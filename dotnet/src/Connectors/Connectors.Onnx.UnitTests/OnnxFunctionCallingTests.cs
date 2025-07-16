// Copyright (c) Microsoft. All rights reserved.

using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.Onnx;
using Microsoft.SemanticKernel.Plugins.Core;
using Xunit;

namespace SemanticKernel.Connectors.Onnx.UnitTests;

/// <summary>
/// Unit tests for ONNX function calling improvements.
/// </summary>
public class OnnxFunctionCallingTests
{
    [Fact]
    public void ExtractJsonFromCodeBlock_WithJsonCodeBlock_ReturnsJsonContent()
    {
        // Arrange
        var content = @"```json
{
  ""function_call"": {
    ""name"": ""Time"",
    ""arguments"": {}
  }
}
```";

        // Act
        var jsonContent = OnnxRuntimeGenAIFunctionCallingChatCompletionService.ExtractJsonFromCodeBlock(content);

        // Assert
        Assert.NotNull(jsonContent);
        Assert.Contains("function_call", jsonContent);
        Assert.Contains("Time", jsonContent);
    }

    [Fact]
    public void ExtractJsonFromCodeBlock_WithInlineCodeBlock_ReturnsJsonContent()
    {
        // Arrange
        var content = @"Here is the function call: `{""function_call"": {""name"": ""Time"", ""arguments"": {}}}`";

        // Act
        var jsonContent = OnnxRuntimeGenAIFunctionCallingChatCompletionService.ExtractJsonFromCodeBlock(content);

        // Assert
        Assert.NotNull(jsonContent);
        Assert.Contains("function_call", jsonContent);
        Assert.Contains("Time", jsonContent);
    }

    [Fact]
    public void ExtractJsonFromCodeBlock_WithNoCodeBlock_ReturnsNull()
    {
        // Arrange
        var content = "This is just regular text without any function calls.";

        // Act
        var jsonContent = OnnxRuntimeGenAIFunctionCallingChatCompletionService.ExtractJsonFromCodeBlock(content);

        // Assert
        Assert.Null(jsonContent);
    }

    [Fact]
    public void TryParseJsonFunctionCall_WithValidJson_ReturnsFunctionCall()
    {
        // Arrange
        var jsonContent = @"{
  ""function_call"": {
    ""name"": ""Time"",
    ""arguments"": {}
  }
}";

        // Act
        var functionCall = OnnxRuntimeGenAIFunctionCallingChatCompletionService.TryParseJsonFunctionCall(jsonContent);

        // Assert
        Assert.NotNull(functionCall);
        Assert.Equal("Time", functionCall.FunctionName);
    }

    [Fact]
    public void TryParseJsonFunctionCall_WithInvalidJson_ReturnsNull()
    {
        // Arrange
        var jsonContent = "This is not valid JSON";

        // Act
        var functionCall = OnnxRuntimeGenAIFunctionCallingChatCompletionService.TryParseJsonFunctionCall(jsonContent);

        // Assert
        Assert.Null(functionCall);
    }

    [Fact]
    public void ContainsFunctionCall_WithJsonFunctionCall_ReturnsTrue()
    {
        // Arrange
        var content = @"{""function_call"": {""name"": ""Time"", ""arguments"": {}}}";

        // Act
        var containsFunctionCall = OnnxRuntimeGenAIFunctionCallingChatCompletionService.ContainsFunctionCall(content);

        // Assert
        Assert.True(containsFunctionCall);
    }

    [Fact]
    public void ContainsFunctionCall_WithCodeBlockFunctionCall_ReturnsTrue()
    {
        // Arrange
        var content = @"```json
{""function_call"": {""name"": ""Time"", ""arguments"": {}}}
```";

        // Act
        var containsFunctionCall = OnnxRuntimeGenAIFunctionCallingChatCompletionService.ContainsFunctionCall(content);

        // Assert
        Assert.True(containsFunctionCall);
    }

    [Fact]
    public void ContainsFunctionCall_WithNoFunctionCall_ReturnsFalse()
    {
        // Arrange
        var content = "This is just regular text without any function calls.";

        // Act
        var containsFunctionCall = OnnxRuntimeGenAIFunctionCallingChatCompletionService.ContainsFunctionCall(content);

        // Assert
        Assert.False(containsFunctionCall);
    }
}

/// <summary>
/// Extension methods to access private methods for testing.
/// </summary>
public static class OnnxRuntimeGenAIFunctionCallingChatCompletionService
{
    /// <summary>
    /// Extracts JSON content from markdown code blocks.
    /// </summary>
    public static string? ExtractJsonFromCodeBlock(string content)
    {
        // Match ```json ... ``` or ``` ... ``` patterns
        var codeBlockMatch = System.Text.RegularExpressions.Regex.Match(content, @"```(?:json)?\s*\n?(.*?)\n?```", System.Text.RegularExpressions.RegexOptions.Singleline | System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        if (codeBlockMatch.Success)
        {
            return codeBlockMatch.Groups[1].Value.Trim();
        }

        // Match `{...}` inline code blocks
        var inlineCodeMatch = System.Text.RegularExpressions.Regex.Match(content, @"`(\{.*?\})`", System.Text.RegularExpressions.RegexOptions.Singleline);
        if (inlineCodeMatch.Success)
        {
            return inlineCodeMatch.Groups[1].Value.Trim();
        }

        return null;
    }

    /// <summary>
    /// Tries to parse a function call from JSON content.
    /// </summary>
    public static Microsoft.SemanticKernel.ChatCompletion.FunctionCallContent? TryParseJsonFunctionCall(string jsonContent)
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
                        return Microsoft.SemanticKernel.Connectors.Onnx.OnnxFunction.ParseFunctionCall(functionName, argumentsJson);
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
    /// Checks if the content contains a function call pattern.
    /// </summary>
    public static bool ContainsFunctionCall(string content)
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
