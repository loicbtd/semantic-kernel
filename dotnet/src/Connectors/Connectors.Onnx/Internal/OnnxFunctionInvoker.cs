// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.SemanticKernel.ChatCompletion;

namespace Microsoft.SemanticKernel.Connectors.Onnx.Internal;

/// <summary>
/// Handles invocation of functions and generation of natural language responses.
/// </summary>
internal sealed class OnnxFunctionInvoker
{
    private readonly ILogger _logger;
    private readonly IChatCompletionService _chatCompletionService;

    public OnnxFunctionInvoker(IChatCompletionService chatCompletionService, ILogger? logger = null)
    {
        this._chatCompletionService = chatCompletionService;
        this._logger = logger ?? NullLogger.Instance;
    }

    /// <summary>
    /// Invokes a function and generates a natural language response.
    /// </summary>
    public async Task<ChatMessageContent> InvokeFunctionAsync(
        FunctionCallContent functionCall,
        string? responseFormat,
        OnnxToolCallingConfig toolCallingConfig,
        ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings,
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
                return new ChatMessageContent(AuthorRole.Assistant, "Function not available.");
            }

            // Try to find the function in the kernel
            KernelFunction? function = FindFunction(functionCall, kernel);
            if (function == null)
            {
                this._logger.LogWarning("Function {FunctionName} not found in any plugin", functionCall.FunctionName);
                return new ChatMessageContent(AuthorRole.Assistant, "Function not found.");
            }

            // Invoke the function with the parsed arguments
            this._logger.LogInformation("Manually invoking function {PluginName}.{FunctionName} with arguments: {Arguments}", 
                function.PluginName, function.Name, 
                functionCall.Arguments != null ? string.Join(", ", functionCall.Arguments.Select(kv => $"{kv.Key}={kv.Value}")) : "none");

            var functionResult = await kernel.InvokeAsync(function, functionCall.Arguments, cancellationToken).ConfigureAwait(false);
            var value = functionResult.GetValue<object>();
            string resultValue = value?.ToString() ?? "<null>";

            // Apply post-processing if response format is specified
            string finalResult;
            if (!string.IsNullOrEmpty(responseFormat))
            {
                this._logger.LogInformation("Applying response format: {ResponseFormat}", responseFormat);
                finalResult = ApplyResponseFormat(responseFormat, resultValue);
                this._logger.LogInformation("Applied response format. Original: {Original}, Final: {Final}", resultValue, finalResult);
            }
            else
            {
                // Return raw result - let the model handle natural language formatting like OpenAI does
                finalResult = resultValue;
            }

            this._logger.LogInformation("Function {PluginName}.{FunctionName} returned: {Result}", function.PluginName, function.Name, resultValue);

            // Generate natural language response
            return await GenerateNaturalResponseAsync(functionCall, function, finalResult, chatHistory, executionSettings, kernel, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            this._logger.LogError(ex, "Manual function invocation failed for {FunctionName}", functionCall.FunctionName);
            return new ChatMessageContent(AuthorRole.Assistant, "Error executing function.");
        }
    }

    /// <summary>
    /// Generates a natural language response based on the function result.
    /// </summary>
    private async Task<ChatMessageContent> GenerateNaturalResponseAsync(
        FunctionCallContent functionCall,
        KernelFunction function,
        string finalResult,
        ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings,
        Kernel kernel,
        CancellationToken cancellationToken)
    {
        // Implement OpenAI-style flow: pass the function result back to model for natural language generation
        ChatHistory tempHistory = [];
        
        // Add a clear system message for natural response generation
        tempHistory.AddSystemMessage("You are a helpful assistant. Answer the user's question directly and naturally based on the function result provided. Be concise and conversational.");
        
        // Get the original user question from the chat history
        string originalQuestion = "";
        foreach (ChatMessageContent message in chatHistory)
        {
            if (message.Role == AuthorRole.User)
            {
                originalQuestion = message.Content ?? "";
            }
        }
        
        // Structure the context more clearly for the model
        tempHistory.AddUserMessage($"The user asked: \"{originalQuestion}\"");
        tempHistory.AddAssistantMessage($"I'll check the {function.Name} function to answer that.");
        tempHistory.Add(new ChatMessageContent(AuthorRole.Tool, $"Function {function.Name} returned: {finalResult}")
        {
            ModelId = functionCall.Id
        });
        tempHistory.AddUserMessage("Based on this function result, what is your answer to my original question?");

        try
        {
            var naturalResponse = await this._chatCompletionService.GetChatMessageContentsAsync(tempHistory, executionSettings, kernel, cancellationToken).ConfigureAwait(false);
            var response = naturalResponse.FirstOrDefault()?.Content ?? finalResult;
            
            this._logger.LogInformation("Model generated natural response: {Response}", response);
            return new ChatMessageContent(AuthorRole.Assistant, response);
        }
        catch (Exception ex)
        {
            this._logger.LogError(ex, "Failed to generate natural language response, returning formatted result");
            return new ChatMessageContent(AuthorRole.Assistant, OnnxResponseFormatter.FormatFunctionResult(function.Name, finalResult));
        }
    }

    /// <summary>
    /// Finds a function in the kernel by name.
    /// </summary>
    private KernelFunction? FindFunction(FunctionCallContent functionCall, Kernel kernel)
    {
        // First try with the provided plugin name
        if (!string.IsNullOrEmpty(functionCall.PluginName))
        {
            var function = kernel.Plugins.GetFunction(functionCall.PluginName, functionCall.FunctionName);
            if (function != null)
            {
                return function;
            }
        }
        
        // If not found, search across all plugins
        foreach (var plugin in kernel.Plugins)
        {
            if (plugin.TryGetFunction(functionCall.FunctionName, out var function))
            {
                this._logger.LogInformation("Found function {FunctionName} in plugin {PluginName}", functionCall.FunctionName, plugin.Name);
                return function;
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

    /// <summary>
    /// Applies response format to the function result.
    /// </summary>
    private static string ApplyResponseFormat(string responseFormat, string functionResult)
    {
        try
        {
            // Replace {result} with the actual function result
            var result = responseFormat.Replace("{result}", functionResult);

            // If the response format doesn't contain {result}, it means the model provided a complete sentence
            // Return it as-is (this is the versatile approach)
            if (!responseFormat.Contains("{result}"))
            {
                return responseFormat;
            }

            return result;
        }
        catch (Exception)
        {
            // If any error occurs during formatting, return the original function result
            return functionResult;
        }
    }
}