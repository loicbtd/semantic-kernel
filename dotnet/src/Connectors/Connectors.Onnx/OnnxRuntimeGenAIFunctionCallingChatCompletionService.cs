// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntimeGenAI;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.FunctionCalling;
using Microsoft.SemanticKernel.Services;

namespace Microsoft.SemanticKernel.Connectors.Onnx;

/// <summary>
/// Represents a chat completion service using OnnxRuntimeGenAI with function calling support.
/// </summary>
public sealed class OnnxRuntimeGenAIFunctionCallingChatCompletionService : IChatCompletionService, IDisposable
{
    private readonly string _modelPath;
    private readonly ILogger _logger;
    private readonly FunctionCallsProcessor _functionCallsProcessor;
    private OnnxRuntimeGenAIChatClient? _chatClient;
    private IChatCompletionService? _chatClientWrapper;
    private readonly Dictionary<string, object?> _attributesInternal = [];

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object?> Attributes => this._attributesInternal;

    /// <summary>
    /// Initializes a new instance of the OnnxRuntimeGenAIFunctionCallingChatCompletionService class.
    /// </summary>
    /// <param name="modelId">The name of the model.</param>
    /// <param name="modelPath">The generative AI ONNX model path for the chat completion service.</param>
    /// <param name="loggerFactory">Optional logger factory to be used for logging.</param>
    /// <param name="jsonSerializerOptions">The <see cref="JsonSerializerOptions"/> to use for various aspects of serialization and deserialization required by the service.</param>
    public OnnxRuntimeGenAIFunctionCallingChatCompletionService(
        string modelId,
        string modelPath,
        ILoggerFactory? loggerFactory = null,
        JsonSerializerOptions? jsonSerializerOptions = null)
    {
        Verify.NotNullOrWhiteSpace(modelId);
        Verify.NotNullOrWhiteSpace(modelPath);

        this._attributesInternal.Add(AIServiceExtensions.ModelIdKey, modelId);
        this._modelPath = modelPath;
        this._logger = loggerFactory?.CreateLogger<OnnxRuntimeGenAIFunctionCallingChatCompletionService>() ?? NullLogger<OnnxRuntimeGenAIFunctionCallingChatCompletionService>.Instance;
        this._functionCallsProcessor = new FunctionCallsProcessor(this._logger);
    }

    private IChatCompletionService GetChatCompletionService()
    {
        this._chatClient ??= new OnnxRuntimeGenAIChatClient(this._modelPath, new OnnxRuntimeGenAIChatClientOptions()
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
    public void Dispose() => this._chatClient?.Dispose();

    /// <inheritdoc/>
    public async Task<IReadOnlyList<ChatMessageContent>> GetChatMessageContentsAsync(
        ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings = null,
        Kernel? kernel = null,
        CancellationToken cancellationToken = default)
    {
        var onnxExecutionSettings = GetOnnxExecutionSettings(executionSettings);
        var toolCallingConfig = GetToolCallingConfig(onnxExecutionSettings, kernel, chatHistory, 0);

        // If no function calling is configured, use the base service
        if (toolCallingConfig is null || toolCallingConfig.Tools is null || toolCallingConfig.Tools.Count == 0)
        {
            return await this.GetChatCompletionService().GetChatMessageContentsAsync(chatHistory, executionSettings, kernel, cancellationToken);
        }

        // Create a modified chat history with function definitions
        var modifiedChatHistory = await this.CreateModifiedChatHistoryAsync(chatHistory, toolCallingConfig, cancellationToken);

        // Get the response from the underlying service
        var results = await this.GetChatCompletionService().GetChatMessageContentsAsync(modifiedChatHistory, executionSettings, kernel, cancellationToken);

        // Process the results for function calls
        return await this.ProcessChatMessageContentsAsync(results, toolCallingConfig, chatHistory, 0, executionSettings, kernel, cancellationToken);
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<StreamingChatMessageContent> GetStreamingChatMessageContentsAsync(
        ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings = null,
        Kernel? kernel = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var onnxExecutionSettings = GetOnnxExecutionSettings(executionSettings);
        var toolCallingConfig = GetToolCallingConfig(onnxExecutionSettings, kernel, chatHistory, 0);

        // If no function calling is configured, use the base service
        if (toolCallingConfig is null || toolCallingConfig.Tools is null || toolCallingConfig.Tools.Count == 0)
        {
            await foreach (var content in this.GetChatCompletionService().GetStreamingChatMessageContentsAsync(chatHistory, executionSettings, kernel, cancellationToken))
            {
                yield return content;
            }
            yield break;
        }

        // Create a modified chat history with function definitions
        var modifiedChatHistory = await this.CreateModifiedChatHistoryAsync(chatHistory, toolCallingConfig, cancellationToken);

        // Get the streaming response from the underlying service
        var responseBuilder = new StringBuilder();
        var streamingResults = new List<StreamingChatMessageContent>();

        await foreach (var content in this.GetChatCompletionService().GetStreamingChatMessageContentsAsync(modifiedChatHistory, executionSettings, kernel, cancellationToken))
        {
            streamingResults.Add(content);
            responseBuilder.Append(content.Content);
            yield return content;
        }

        // Process the complete response for function calls
        var completeResponse = new ChatMessageContent(AuthorRole.Assistant, responseBuilder.ToString());
        await this.ProcessSingleChatMessageForFunctionCallsAsync(completeResponse, toolCallingConfig, chatHistory, 0, executionSettings, kernel, cancellationToken);
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
        if (executionSettings.ToolCallBehavior is null)
        {
            return null;
        }

        return executionSettings.ToolCallBehavior.ConfigureRequest(kernel, chatHistory, requestIndex);
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
            var systemMessage = $"You are a helpful assistant with access to the following functions. Use them when appropriate:\n\n{functionsJson}\n\nWhen you want to call a function, respond with JSON in the format: {{\"function_call\": {{\"name\": \"function_name\", \"arguments\": {{\"param1\": \"value1\"}}}}}}\n\nDo not include any other text when making a function call.";
            
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

        return functionsArray.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
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
        var processedResults = new List<ChatMessageContent>();

        foreach (var result in results)
        {
            var processedResult = await this.ProcessSingleChatMessageForFunctionCallsAsync(
                result, toolCallingConfig, chatHistory, requestIndex, executionSettings, kernel, cancellationToken);
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

        // Try to parse function call from the response
        var functionCall = TryParseFunctionCall(result.Content);
        if (functionCall is null)
        {
            return result;
        }

        // Add function call to the result
        result.Items.Add(functionCall);

        // Process function calls if auto-invoke is enabled
        var lastMessage = await this._functionCallsProcessor.ProcessFunctionCallsAsync(
            result,
            executionSettings,
            chatHistory,
            requestIndex,
            (functionCallContent) => IsValidFunctionCall(functionCallContent, toolCallingConfig),
            toolCallingConfig.Options ?? new FunctionChoiceBehaviorOptions(),
            kernel,
            isStreaming: false,
            cancellationToken);

        return lastMessage ?? result;
    }

    /// <summary>
    /// Tries to parse a function call from the model's response.
    /// </summary>
    private static FunctionCallContent? TryParseFunctionCall(string? content)
    {
        if (string.IsNullOrEmpty(content))
        {
            return null;
        }

        try
        {
            // Try to parse as JSON
            var jsonDocument = JsonDocument.Parse(content);
            if (jsonDocument.RootElement.TryGetProperty("function_call", out var functionCallElement))
            {
                if (functionCallElement.TryGetProperty("name", out var nameElement) &&
                    functionCallElement.TryGetProperty("arguments", out var argumentsElement))
                {
                    var functionName = nameElement.GetString();
                    var argumentsJson = argumentsElement.GetRawText();

                    if (!string.IsNullOrEmpty(functionName))
                    {
                        return OnnxFunction.ParseFunctionCall(functionName, argumentsJson);
                    }
                }
            }
        }
        catch (JsonException)
        {
            // If JSON parsing fails, try to extract function call using regex
            var functionCallMatch = Regex.Match(content, @"(?:function_call|call|invoke)\s*:?\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)", RegexOptions.IgnoreCase);
            if (functionCallMatch.Success)
            {
                var functionName = functionCallMatch.Groups[1].Value;
                var argumentsString = functionCallMatch.Groups[2].Value;

                return OnnxFunction.ParseFunctionCall(functionName, argumentsString);
            }
        }

        return null;
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