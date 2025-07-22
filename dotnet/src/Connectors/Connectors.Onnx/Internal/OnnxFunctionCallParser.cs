// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.SemanticKernel.ChatCompletion;

namespace Microsoft.SemanticKernel.Connectors.Onnx.Internal;

/// <summary>
/// Handles parsing of function calls from model responses.
/// </summary>
internal sealed class OnnxFunctionCallParser
{
    private readonly ILogger _logger;

    public OnnxFunctionCallParser(ILogger? logger = null)
    {
        this._logger = logger ?? NullLogger.Instance;
    }

    /// <summary>
    /// Checks if the content contains a function call pattern.
    /// </summary>
    public bool ContainsFunctionCall(string content)
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

    /// <summary>
    /// Tries to parse a function call with response format from the model's response.
    /// </summary>
    public (FunctionCallContent? FunctionCall, string? ResponseFormat)? TryParseFunctionCallWithResponse(string? content)
    {
        if (string.IsNullOrEmpty(content))
        {
            return null;
        }

        this._logger.LogInformation("Attempting to parse function call from content: {Content}", content);

        // First, try to parse the entire content as pure JSON with response format
        var resultFromContent = TryParseJsonFunctionCallWithResponse(content.Trim());
        if (resultFromContent != null)
        {
            this._logger.LogInformation("Successfully parsed function call from pure JSON content: {FunctionName}", resultFromContent.Value.FunctionCall?.FunctionName);
            return resultFromContent;
        }

        // --- Robustness: Try to fix missing closing brace ---
        var trimmed = content.Trim();
        if (trimmed.StartsWith("{") && trimmed.Contains("function_call") && !trimmed.EndsWith("}"))
        {
            var fixedJson = trimmed + "}";
            var fixedResult = TryParseJsonFunctionCallWithResponse(fixedJson);
            if (fixedResult != null)
            {
                this._logger.LogInformation("Successfully parsed function call from auto-corrected JSON (added closing brace)");
                return fixedResult;
            }
        }
        // -----------------------------------------------------

        // Try to extract JSON from the first line and natural language from the rest
        var lines = content.Split('\n');
        if (lines.Length >= 2)
        {
            var firstLine = lines[0].Trim();
            var remainingText = string.Join("\n", lines.Skip(1)).Trim();

            // Check if first line is JSON with empty function_call
            var jsonResult = TryParseJsonFunctionCallWithResponse(firstLine);
            if (jsonResult != null && jsonResult.Value.FunctionCall == null)
            {
                // Empty function_call with natural language response
                this._logger.LogInformation("Successfully parsed empty function call with natural language response");
                return (null, remainingText);
            }
        }

        // If not pure JSON, try to extract JSON from code blocks (```json ... ```)
        var jsonContent = ExtractJsonFromCodeBlock(content);
        if (!string.IsNullOrEmpty(jsonContent))
        {
            this._logger.LogInformation("Extracted JSON from code block: {JsonContent}", jsonContent);
            var result = TryParseJsonFunctionCallWithResponse(jsonContent);
            if (result != null)
            {
                this._logger.LogInformation("Successfully parsed function call from code block: {FunctionName}", result.Value.FunctionCall?.FunctionName);
                return result;
            }
        }

        // Fallback: try to parse without response format
        var functionCallFromContent = TryParseJsonFunctionCall(content.Trim());
        if (functionCallFromContent != null)
        {
            this._logger.LogInformation("Successfully parsed function call from pure JSON content (no response format): {FunctionName}", functionCallFromContent.FunctionName);
            return (functionCallFromContent, null);
        }

        // If JSON parsing fails, try to extract function call using regex
        var functionCallMatch = Regex.Match(content, @"(?:function_call|call|invoke)\s*:?\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)",
            RegexOptions.IgnoreCase);
        if (functionCallMatch.Success)
        {
            var functionName = functionCallMatch.Groups[1].Value;
            var argumentsString = functionCallMatch.Groups[2].Value;

            this._logger.LogInformation("Successfully parsed function call using regex: {FunctionName}", functionName);
            var functionCall = OnnxFunction.ParseFunctionCall(functionName, argumentsString);
            return (functionCall, null);
        }

        this._logger.LogInformation("Failed to parse function call from content");
        return null;
    }

    /// <summary>
    /// Tries to parse a function call from the model's response (legacy method for backward compatibility).
    /// </summary>
    public FunctionCallContent? TryParseFunctionCall(string? content)
    {
        var result = TryParseFunctionCallWithResponse(content);
        return result?.FunctionCall;
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
                if (functionCallElement.TryGetProperty("name", out var nameElement))
                {
                    var functionName = nameElement.GetString();
                    
                    if (!string.IsNullOrEmpty(functionName))
                    {
                        // Try to get arguments, use empty object if not present
                        var argumentsJson = "{}";
                        if (functionCallElement.TryGetProperty("arguments", out var argumentsElement))
                        {
                            argumentsJson = argumentsElement.GetRawText();
                        }
                        
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
    /// Tries to parse a function call with response format from JSON content.
    /// </summary>
    private static (FunctionCallContent? FunctionCall, string? ResponseFormat)? TryParseJsonFunctionCallWithResponse(string jsonContent)
    {
        try
        {
            var jsonDocument = JsonDocument.Parse(jsonContent);
            if (jsonDocument.RootElement.TryGetProperty("function_call", out var functionCallElement))
            {
                // Check for response_format first
                string? responseFormat = null;
                if (jsonDocument.RootElement.TryGetProperty("response_format", out var responseFormatElement))
                {
                    responseFormat = responseFormatElement.GetString();
                }

                // Check if function_call is a string (direct function name)
                if (functionCallElement.ValueKind == JsonValueKind.String)
                {
                    var functionName = functionCallElement.GetString();
                    if (!string.IsNullOrEmpty(functionName))
                    {
                        var functionCall = OnnxFunction.ParseFunctionCall(functionName, "{}");
                        return (functionCall, responseFormat);
                    }
                }

                // Check if this is a natural language response (empty function_call)
                if (functionCallElement.ValueKind == JsonValueKind.Object &&
                    functionCallElement.EnumerateObject().Count() == 0)
                {
                    // Empty function_call means natural language response
                    return (null, responseFormat);
                }

                // Check if this is an actual function call with name (arguments are optional)
                if (functionCallElement.TryGetProperty("name", out var nameElement))
                {
                    var functionName = nameElement.GetString();
                    
                    if (!string.IsNullOrEmpty(functionName))
                    {
                        // Try to get arguments, use empty object if not present
                        var argumentsJson = "{}";
                        if (functionCallElement.TryGetProperty("arguments", out var argumentsElement))
                        {
                            argumentsJson = argumentsElement.GetRawText();
                        }
                        
                        var functionCall = OnnxFunction.ParseFunctionCall(functionName, argumentsJson);
                        return (functionCall, responseFormat);
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
}