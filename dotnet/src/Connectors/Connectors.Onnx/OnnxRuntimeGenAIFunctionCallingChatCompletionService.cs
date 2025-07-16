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
        
        this._logger.LogInformation("OnnxRuntimeGenAIFunctionCallingChatCompletionService initialized with model: {ModelId} at path: {ModelPath}", modelId, modelPath);
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

        // Return a wrapper that handles the basic chat completion without function calling
        return this._chatClientWrapper ??= new BasicOnnxChatCompletionService(this._chatClient);
    }

    /// <summary>
    /// Basic chat completion service wrapper that doesn't handle function calling.
    /// </summary>
    private sealed class BasicOnnxChatCompletionService : IChatCompletionService
    {
        private readonly OnnxRuntimeGenAIChatClient _chatClient;
        private readonly IChatCompletionService _wrappedService;

        public BasicOnnxChatCompletionService(OnnxRuntimeGenAIChatClient chatClient)
        {
            this._chatClient = chatClient;
            this._wrappedService = chatClient.AsChatCompletionService();
        }

        public IReadOnlyDictionary<string, object?> Attributes => this._wrappedService.Attributes;

        public Task<IReadOnlyList<ChatMessageContent>> GetChatMessageContentsAsync(
            ChatHistory chatHistory,
            PromptExecutionSettings? executionSettings = null,
            Kernel? kernel = null,
            CancellationToken cancellationToken = default)
        {
            return this._wrappedService.GetChatMessageContentsAsync(chatHistory, executionSettings, kernel, cancellationToken);
        }

        public IAsyncEnumerable<StreamingChatMessageContent> GetStreamingChatMessageContentsAsync(
            ChatHistory chatHistory,
            PromptExecutionSettings? executionSettings = null,
            Kernel? kernel = null,
            CancellationToken cancellationToken = default)
        {
            return this._wrappedService.GetStreamingChatMessageContentsAsync(chatHistory, executionSettings, kernel, cancellationToken);
        }
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
        this._logger.LogInformation("GetChatMessageContentsAsync called");
        
        var onnxExecutionSettings = GetOnnxExecutionSettings(executionSettings);
        var toolCallingConfig = GetToolCallingConfig(onnxExecutionSettings, kernel, chatHistory, 0);

        this._logger.LogInformation("Tool calling config: {Config}", toolCallingConfig != null ? "configured" : "null");
        if (toolCallingConfig != null)
        {
            this._logger.LogInformation("Tools count: {Count}, AutoInvoke: {AutoInvoke}", toolCallingConfig.Tools?.Count ?? 0, toolCallingConfig.AutoInvoke);
        }

        // If no function calling is configured, use the base service
        if (toolCallingConfig is null || toolCallingConfig.Tools is null || toolCallingConfig.Tools.Count == 0)
        {
            this._logger.LogInformation("No function calling configured, using base service");
            return await this.GetChatCompletionService().GetChatMessageContentsAsync(chatHistory, executionSettings, kernel, cancellationToken).ConfigureAwait(false);
        }

        this._logger.LogInformation("Function calling is configured, processing with function calling");

        // Create a modified chat history with function definitions
        var modifiedChatHistory = await this.CreateModifiedChatHistoryAsync(chatHistory, toolCallingConfig, cancellationToken).ConfigureAwait(false);

        // Get the response from the underlying service
        var results = await this.GetChatCompletionService().GetChatMessageContentsAsync(modifiedChatHistory, executionSettings, kernel, cancellationToken).ConfigureAwait(false);

        this._logger.LogInformation("Got {Count} results from base service", results.Count);
        foreach (var result in results)
        {
            this._logger.LogInformation("Result content: {Content}", result.Content);
        }

        // Process the results for function calls
        var processedResults = await this.ProcessChatMessageContentsAsync(results, toolCallingConfig, chatHistory, 0, executionSettings, kernel, cancellationToken).ConfigureAwait(false);
        
        this._logger.LogInformation("Processed {Count} results", processedResults.Count);
        foreach (var result in processedResults)
        {
            this._logger.LogInformation("Processed result content: {Content}", result.Content);
        }

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
            if (!hasFunctionCall && ContainsFunctionCall(currentResponse))
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

        // Try to parse function call from the response
        var functionCall = TryParseFunctionCall(result.Content);
        if (functionCall is null)
        {
            return result;
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
            if (lastMessage != null && !ContainsFunctionCall(lastMessage.Content))
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
        return await TryManualFunctionInvocationAsync(result, functionCall, toolCallingConfig, kernel, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Manual fallback for function invocation when standard processing fails.
    /// </summary>
    private async Task<ChatMessageContent> TryManualFunctionInvocationAsync(
        ChatMessageContent result,
        FunctionCallContent functionCall,
        OnnxToolCallingConfig toolCallingConfig,
        Kernel kernel,
        CancellationToken cancellationToken)
    {
        this._logger.LogInformation("Manual fallback invoked for function: {FunctionName}", functionCall.FunctionName);
        
        try
        {
            // Check if the function call is valid
            if (!IsValidFunctionCall(functionCall, toolCallingConfig))
            {
                this._logger.LogWarning("Function call {FunctionName} is not in the allowed functions list", functionCall.FunctionName);
                return result;
            }

            // Try to find the function in the kernel
            var function = kernel.Plugins.GetFunction(functionCall.PluginName ?? "DefaultPlugin", functionCall.FunctionName);
            if (function == null)
            {
                this._logger.LogWarning("Function {FunctionName} not found in kernel", functionCall.FunctionName);
                return result;
            }

            // Invoke the function
            this._logger.LogInformation("Manually invoking function {PluginName}.{FunctionName}", function.PluginName, function.Name);
            
            var functionResult = await kernel.InvokeAsync(function, arguments: null, cancellationToken).ConfigureAwait(false);
            var value = functionResult.GetValue<object>();
            string resultValue = value?.ToString() ?? "<null>";

            // Create a new message with the function result
            var functionResultMessage = new ChatMessageContent(AuthorRole.Assistant, resultValue ?? string.Empty);
            
            this._logger.LogInformation("Function {PluginName}.{FunctionName} returned: {Result}", function.PluginName, function.Name, resultValue);
            this._logger.LogInformation("Manual fallback returning message: {Message}", functionResultMessage.Content);
            
            return functionResultMessage;
        }
        catch (Exception ex)
        {
            this._logger.LogError(ex, "Manual function invocation failed for {FunctionName}", functionCall.FunctionName);
            return result;
        }
    }

    /// <summary>
    /// Tries to parse a function call from the model's response.
    /// </summary>
    private FunctionCallContent? TryParseFunctionCall(string? content)
    {
        if (string.IsNullOrEmpty(content))
        {
            return null;
        }

        this._logger.LogInformation("Attempting to parse function call from content: {Content}", content);

        // First, try to extract JSON from code blocks (```json ... ```)
        var jsonContent = ExtractJsonFromCodeBlock(content);
        if (!string.IsNullOrEmpty(jsonContent))
        {
            this._logger.LogInformation("Extracted JSON from code block: {JsonContent}", jsonContent);
            var functionCall = TryParseJsonFunctionCall(jsonContent);
            if (functionCall != null)
            {
                this._logger.LogInformation("Successfully parsed function call from code block: {FunctionName}", functionCall.FunctionName);
                return functionCall;
            }
        }

        // Try to parse the entire content as JSON
        var functionCallFromContent = TryParseJsonFunctionCall(content);
        if (functionCallFromContent != null)
        {
            this._logger.LogInformation("Successfully parsed function call from content: {FunctionName}", functionCallFromContent.FunctionName);
            return functionCallFromContent;
        }

        // If JSON parsing fails, try to extract function call using regex
        var functionCallMatch = Regex.Match(content, @"(?:function_call|call|invoke)\s*:?\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)", RegexOptions.IgnoreCase);
        if (functionCallMatch.Success)
        {
            var functionName = functionCallMatch.Groups[1].Value;
            var argumentsString = functionCallMatch.Groups[2].Value;

            this._logger.LogInformation("Successfully parsed function call using regex: {FunctionName}", functionName);
            return OnnxFunction.ParseFunctionCall(functionName, argumentsString);
        }

        this._logger.LogInformation("Failed to parse function call from content");
        return null;
    }

    /// <summary>
    /// Extracts JSON content from markdown code blocks.
    /// </summary>
    private static string? ExtractJsonFromCodeBlock(string content)
    {
        // Match ```json ... ``` or ``` ... ``` patterns
        var codeBlockMatch = Regex.Match(content, @"```(?:json)?\s*\n?(.*?)\n?```", RegexOptions.Singleline | RegexOptions.IgnoreCase);
        if (codeBlockMatch.Success)
        {
            return codeBlockMatch.Groups[1].Value.Trim();
        }

        // Match `{...}` inline code blocks
        var inlineCodeMatch = Regex.Match(content, @"`(\{.*?\})`", RegexOptions.Singleline);
        if (inlineCodeMatch.Success)
        {
            return inlineCodeMatch.Groups[1].Value.Trim();
        }

        return null;
    }

    /// <summary>
    /// Tries to parse a function call from JSON content.
    /// </summary>
    private static FunctionCallContent? TryParseJsonFunctionCall(string jsonContent)
    {
        try
        {
            var jsonDocument = JsonDocument.Parse(jsonContent);
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
            // JSON parsing failed, return null to try other methods
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

    /// <summary>
    /// Checks if the content contains a function call pattern.
    /// </summary>
    private static bool ContainsFunctionCall(string content)
    {
        if (string.IsNullOrEmpty(content))
        {
            return false;
        }

        // Check for JSON function call pattern
        if (content.Contains("function_call") && content.Contains("{") && content.Contains("}"))
        {
            return true;
        }

        // Check for code block with function call
        if (content.Contains("```") && content.Contains("function_call"))
        {
            return true;
        }

        return false;
    }
}
