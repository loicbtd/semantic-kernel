// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization.Metadata;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntimeGenAI;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.FunctionCalling;
using Microsoft.SemanticKernel.Connectors.Onnx.Internal;
using Microsoft.SemanticKernel.Services;

namespace Microsoft.SemanticKernel.Connectors.Onnx;

/// <summary>
/// Represents a chat completion service using OnnxRuntimeGenAI with function calling support.
/// </summary>
public sealed class OnnxRuntimeGenAIFunctionCallingChatCompletionService : IChatCompletionService, IDisposable
{
    private readonly Config _config;
    private readonly Model _model;
    private readonly ILogger _logger;
    private readonly FunctionCallsProcessor _functionCallsProcessor;
    private readonly OnnxFunctionCallParser _functionCallParser;
    private OnnxFunctionInvoker? _functionInvoker;
    private OnnxRuntimeGenAIChatClient? _chatClient;
    private IChatCompletionService? _chatClientWrapper;
    private readonly Dictionary<string, object?> _attributesInternal = [];

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object?> Attributes => this._attributesInternal;

    /// <summary>
    /// Initializes a new instance of the OnnxRuntimeGenAIFunctionCallingChatCompletionService class.
    /// </summary>
    /// <param name="modelPath">The generative AI ONNX model path for the chat completion service.</param>
    /// <param name="modelId">The name of the model.</param>
    /// <param name="loggerFactory">Optional logger factory to be used for logging.</param>
    /// <param name="jsonSerializerOptions">The <see cref="JsonSerializerOptions"/> to use for various aspects of serialization and deserialization required by the service.</param>
    /// <param name="providers">The providers to use for the chat completion service.</param>
    public OnnxRuntimeGenAIFunctionCallingChatCompletionService(
        string modelPath,
        string? modelId = null,
        ILoggerFactory? loggerFactory = null,
        JsonSerializerOptions? jsonSerializerOptions = null,
        List<string>? providers = null)
    {
        Verify.NotNullOrWhiteSpace(modelPath);
        this._attributesInternal.Add(AIServiceExtensions.ModelIdKey, modelId ?? Path.GetFileName(modelPath));
        this._config = new Config(modelPath);
        if (providers != null)
        {
            this._config.ClearProviders();
            foreach (string provider in providers)
            {
                this._config.AppendProvider(provider);
            }
        }

        this._model = new Model(this._config);

        this._logger = loggerFactory?.CreateLogger<OnnxRuntimeGenAIFunctionCallingChatCompletionService>() ?? NullLogger<OnnxRuntimeGenAIFunctionCallingChatCompletionService>.Instance;
        this._functionCallsProcessor = new FunctionCallsProcessor(this._logger);
        this._functionCallParser = new OnnxFunctionCallParser(this._logger);
    }

    private IChatCompletionService GetChatCompletionService()
    {
        this._chatClient ??= new OnnxRuntimeGenAIChatClient(this._model, false, new OnnxRuntimeGenAIChatClientOptions
        {
            PromptFormatter = (messages, options) =>
            {
                StringBuilder promptBuilder = new();
                foreach (var message in messages)
                {
                    promptBuilder.Append($"<|{message.Role}|>\n{message.Text}");
                }

                promptBuilder.Append("<|end|>\n<|assistant|>");

                return promptBuilder.ToString();
            }
        });

        return this._chatClientWrapper ??= this._chatClient.AsChatCompletionService();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        this._model.Dispose();
        this._config.Dispose();
        this._chatClient?.Dispose();
    }

    /// <inheritdoc/>
    public async Task<IReadOnlyList<ChatMessageContent>> GetChatMessageContentsAsync(
        ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings = null,
        Kernel? kernel = null,
        CancellationToken cancellationToken = default
    )
    {
        OnnxRuntimeGenAIPromptExecutionSettings onnxExecutionSettings = GetOnnxExecutionSettings(executionSettings);

        OnnxToolCallingConfig? toolCallingConfig = this.GetToolCallingConfig(onnxExecutionSettings, kernel, chatHistory, 0);
        if (toolCallingConfig?.Tools is null || toolCallingConfig.Tools.Count == 0)
        {
            return await this.GetChatCompletionService().GetChatMessageContentsAsync(chatHistory, executionSettings, kernel, cancellationToken).ConfigureAwait(false);
        }

        // Create a modified chat history with function definitions
        ChatHistory modifiedChatHistory = await this.CreateModifiedChatHistoryAsync(chatHistory, toolCallingConfig, cancellationToken).ConfigureAwait(false);

        // Get the response from the underlying service
        IReadOnlyList<ChatMessageContent> results = await this.GetChatCompletionService().GetChatMessageContentsAsync(modifiedChatHistory, executionSettings, kernel, cancellationToken).ConfigureAwait(false);

        IReadOnlyList<ChatMessageContent> processedResults = await this.ProcessChatMessageContentsAsync(results, toolCallingConfig, chatHistory, 0, executionSettings, kernel, cancellationToken).ConfigureAwait(false);
        return processedResults;
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<StreamingChatMessageContent> GetStreamingChatMessageContentsAsync(
        ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings = null,
        Kernel? kernel = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation]
        CancellationToken cancellationToken = default)
    {
        var onnxExecutionSettings = GetOnnxExecutionSettings(executionSettings);
        var toolCallingConfig = GetToolCallingConfig(onnxExecutionSettings, kernel, chatHistory, 0);

        // If no function calling is configured, use the base service
        if (toolCallingConfig is null || toolCallingConfig.Tools is null || toolCallingConfig.Tools.Count == 0)
        {
            await foreach (var content in this.GetChatCompletionService().GetStreamingChatMessageContentsAsync(chatHistory, executionSettings, kernel, cancellationToken).ConfigureAwait(false))
            {
                yield return content;
            }

            yield break;
        }

        // Create a modified chat history with function definitions
        var modifiedChatHistory = await this.CreateModifiedChatHistoryAsync(chatHistory, toolCallingConfig, cancellationToken).ConfigureAwait(false);

        // Get the streaming response from the underlying service
        var responseBuilder = new StringBuilder();
        var hasFunctionCall = false;
        var functionCallContent = new List<StreamingChatMessageContent>();

        await foreach (var content in this.GetChatCompletionService().GetStreamingChatMessageContentsAsync(modifiedChatHistory, executionSettings, kernel, cancellationToken).ConfigureAwait(false))
        {
            responseBuilder.Append(content.Content);
            functionCallContent.Add(content);

            // Check if we have a complete function call
            var currentResponse = responseBuilder.ToString();
            if (!hasFunctionCall && this._functionCallParser.ContainsFunctionCall(currentResponse))
            {
                hasFunctionCall = true;
                this._logger.LogInformation("Detected function call in streaming response");
            }

            // If we have a function call, don't yield the content yet
            if (!hasFunctionCall)
            {
                yield return content;
            }
        }

        // If we detected a function call, process it and yield the result
        if (hasFunctionCall)
        {
            var completeResponse = new ChatMessageContent(AuthorRole.Assistant, responseBuilder.ToString());
            var processedResponse = await this.ProcessSingleChatMessageForFunctionCallsAsync(completeResponse, toolCallingConfig, chatHistory, 0, executionSettings, kernel, cancellationToken).ConfigureAwait(false);

            // Yield the processed response instead of the original JSON
            if (processedResponse.Content != completeResponse.Content)
            {
                // Split the response into chunks for streaming effect
                var words = processedResponse.Content.Split(' ');
                foreach (var word in words)
                {
                    yield return new StreamingChatMessageContent(AuthorRole.Assistant, word + " ");
                    await Task.Delay(50, cancellationToken).ConfigureAwait(false); // Small delay for streaming effect
                }
            }
            else
            {
                // If processing didn't change the content, yield the original
                foreach (var content in functionCallContent)
                {
                    yield return content;
                }
            }
        }
    }

    /// <summary>
    /// Gets the ONNX execution settings from the provided execution settings.
    /// </summary>
    private static OnnxRuntimeGenAIPromptExecutionSettings GetOnnxExecutionSettings(PromptExecutionSettings? executionSettings)
    {
        return executionSettings switch
        {
            null => new OnnxRuntimeGenAIPromptExecutionSettings(),
            OnnxRuntimeGenAIPromptExecutionSettings onnxSettings => onnxSettings,
            _ => OnnxRuntimeGenAIPromptExecutionSettings.FromExecutionSettings(executionSettings)
        };
    }

    /// <summary>
    /// Gets the tool calling configuration from the execution settings.
    /// </summary>
    private OnnxToolCallingConfig? GetToolCallingConfig(
        OnnxRuntimeGenAIPromptExecutionSettings executionSettings,
        Kernel? kernel,
        ChatHistory chatHistory,
        int requestIndex)
    {
        // First try ToolCallBehavior (ONNX-specific)
        if (executionSettings.ToolCallBehavior is not null)
        {
            return executionSettings.ToolCallBehavior.ConfigureRequest(kernel, chatHistory, requestIndex);
        }

        // Then try FunctionChoiceBehavior (standard Semantic Kernel)
        if (executionSettings.FunctionChoiceBehavior is not null)
        {
            var config = this._functionCallsProcessor.GetConfiguration(
                executionSettings.FunctionChoiceBehavior,
                chatHistory,
                requestIndex,
                kernel);

            if (config is not null)
            {
                // Convert to OnnxToolCallingConfig
                return new OnnxToolCallingConfig(
                    Tools: config.Functions?.Select(f => new OnnxFunction(
                        f.Metadata.Name,
                        f.Metadata.Description,
                        f.Metadata.Parameters,
                        f.Metadata.ReturnParameter)).ToList(),
                    AutoInvoke: config.AutoInvoke,
                    AllowAnyRequestedKernelFunction: false, // Standard behavior doesn't allow unrequested functions
                    Options: config.Options);
            }
        }

        return null;
    }

    /// <summary>
    /// Creates a modified chat history with function definitions included.
    /// </summary>
    private async Task<ChatHistory> CreateModifiedChatHistoryAsync(
        ChatHistory originalChatHistory,
        OnnxToolCallingConfig toolCallingConfig,
        CancellationToken cancellationToken)
    {
        var modifiedChatHistory = new ChatHistory();

        // Add function definitions as a system message
        if (toolCallingConfig.Tools is not null && toolCallingConfig.Tools.Count > 0)
        {
            var functionsJson = CreateFunctionsJson(toolCallingConfig.Tools);
            var systemMessage = "You are a helpful assistant with access to the following functions:"
                                + functionsJson +
                                """

                                IMPORTANT: You MUST use these functions when the user's request relates to them. Do NOT provide general responses when a function is available.

                                For ANY user question that can be answered by calling a function, respond with raw JSON (no markdown formatting):
                                {"function_call": {"name": "function_name", "arguments": {"param1": "value1"}}}

                                For natural language responses after function results, respond with JSON:
                                {"function_call": {}, "response_format": "your response here"}

                                Return only the JSON object, without code blocks or markdown formatting. ALWAYS call the relevant function first. NEVER give general explanations when a function can handle the request.
                                """;

            modifiedChatHistory.AddSystemMessage(systemMessage);
        }

        // Add original chat history
        foreach (var message in originalChatHistory)
        {
            modifiedChatHistory.Add(message);
        }

        return modifiedChatHistory;
    }

    /// <summary>
    /// Creates a JSON representation of the available functions.
    /// </summary>
    private static string CreateFunctionsJson(IList<OnnxFunction> functions)
    {
        var functionsArray = new JsonArray();

        foreach (var function in functions)
        {
            functionsArray.Add(function.ToFunctionDefinition());
        }

        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            TypeInfoResolver = new DefaultJsonTypeInfoResolver()
        };
        options.MakeReadOnly();

        return functionsArray.ToJsonString(options);
    }

    /// <summary>
    /// Processes chat message contents for function calls.
    /// </summary>
    private async Task<IReadOnlyList<ChatMessageContent>> ProcessChatMessageContentsAsync(
        IReadOnlyList<ChatMessageContent> results,
        OnnxToolCallingConfig toolCallingConfig,
        ChatHistory chatHistory,
        int requestIndex,
        PromptExecutionSettings? executionSettings,
        Kernel? kernel,
        CancellationToken cancellationToken)
    {
        this._logger.LogInformation("ProcessChatMessageContentsAsync called with {Count} results", results.Count);

        var processedResults = new List<ChatMessageContent>();

        foreach (var result in results)
        {
            this._logger.LogInformation("Processing result: {Content}", result.Content);
            var processedResult = await this.ProcessSingleChatMessageForFunctionCallsAsync(result, toolCallingConfig, chatHistory, requestIndex, executionSettings, kernel, cancellationToken).ConfigureAwait(false);
            this._logger.LogInformation("Processed result: {Content}", processedResult.Content);
            processedResults.Add(processedResult);
        }

        return processedResults;
    }

    /// <summary>
    /// Processes a single chat message for function calls.
    /// </summary>
    private async Task<ChatMessageContent> ProcessSingleChatMessageForFunctionCallsAsync(
        ChatMessageContent result,
        OnnxToolCallingConfig toolCallingConfig,
        ChatHistory chatHistory,
        int requestIndex,
        PromptExecutionSettings? executionSettings,
        Kernel? kernel,
        CancellationToken cancellationToken)
    {
        if (!toolCallingConfig.AutoInvoke || kernel is null)
        {
            return result;
        }

        // Quick pre-check: if content doesn't contain function call patterns, skip parsing
        if (!this._functionCallParser.ContainsFunctionCall(result.Content))
        {
            this._logger.LogDebug("Content does not contain function call patterns, skipping parsing");
            return result;
        }

        // Try to parse function call from the response
        var parseResult = this._functionCallParser.TryParseFunctionCallWithResponse(result.Content);
        if (parseResult is null)
        {
            return result;
        }

        var functionCall = parseResult.Value.FunctionCall;
        var responseFormat = parseResult.Value.ResponseFormat;

        // If functionCall is null, this is a natural language response
        if (functionCall == null)
        {
            this._logger.LogInformation("Detected natural language response with response_format: {ResponseFormat}", responseFormat);

            // Create a new message with the response format content
            var newContent = !string.IsNullOrEmpty(responseFormat) ? responseFormat : result.Content;
            return new ChatMessageContent(AuthorRole.Assistant, newContent);
        }

        // Add function call to the result
        result.Items.Add(functionCall);

        try
        {
            // Try the standard function calls processor first
            this._logger.LogInformation("Attempting standard function call processing");
            var lastMessage = await this._functionCallsProcessor.ProcessFunctionCallsAsync(
                result,
                executionSettings,
                chatHistory,
                requestIndex,
                (functionCallContent) => IsValidFunctionCall(functionCallContent, toolCallingConfig),
                toolCallingConfig.Options ?? new FunctionChoiceBehaviorOptions(),
                kernel,
                isStreaming: false,
                cancellationToken).ConfigureAwait(false);

            this._logger.LogInformation("Standard processing returned message: {Content}", lastMessage?.Content ?? "null");

            // Check if the standard processing actually replaced the JSON content
            if (lastMessage != null && !this._functionCallParser.ContainsFunctionCall(lastMessage.Content))
            {
                this._logger.LogInformation("Standard function call processing succeeded, content replaced");
                return lastMessage;
            }
            else
            {
                this._logger.LogInformation("Standard function call processing returned JSON content or null, using manual fallback");
            }
        }
        catch (Exception ex)
        {
            this._logger.LogWarning(ex, "Standard function call processing failed, trying manual fallback");
        }

        // Fallback: Manual function invocation if standard processing fails or returns JSON
        this._logger.LogInformation("Using manual fallback for function call");
        this._functionInvoker ??= new OnnxFunctionInvoker(this.GetChatCompletionService(), this._logger);
        return await this._functionInvoker.InvokeFunctionAsync(functionCall, responseFormat, toolCallingConfig, chatHistory, executionSettings, kernel, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Checks if a function call is valid (i.e., the function was advertised to the model).
    /// </summary>
    private static bool IsValidFunctionCall(FunctionCallContent functionCallContent, OnnxToolCallingConfig toolCallingConfig)
    {
        if (toolCallingConfig.AllowAnyRequestedKernelFunction)
        {
            return true;
        }

        if (toolCallingConfig.Tools is null)
        {
            return false;
        }

        return toolCallingConfig.Tools.Any(tool => tool.FunctionName == functionCallContent.FunctionName);
    }
}
