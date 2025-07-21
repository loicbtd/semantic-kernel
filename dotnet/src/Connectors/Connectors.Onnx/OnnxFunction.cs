// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using Microsoft.SemanticKernel.ChatCompletion;

namespace Microsoft.SemanticKernel.Connectors.Onnx;

/// <summary>
/// Represents a function that can be called by the ONNX model.
/// </summary>
[JsonConverter(typeof(OnnxFunctionJsonConverter))]
public sealed class OnnxFunction
{
    /// <summary>
    /// Initializes a new instance of the <see cref="OnnxFunction"/> class.
    /// </summary>
    /// <param name="functionName">The name of the function.</param>
    /// <param name="description">The description of the function.</param>
    /// <param name="parameters">The parameters of the function.</param>
    /// <param name="returnParameter">The return parameter of the function.</param>
    public OnnxFunction(string functionName, string description, IEnumerable<KernelParameterMetadata>? parameters = null, KernelReturnParameterMetadata? returnParameter = null)
    {
        Verify.NotNullOrWhiteSpace(functionName, nameof(functionName));
        Verify.NotNullOrWhiteSpace(description, nameof(description));

        this.FunctionName = functionName;
        this.Description = description;
        this.Parameters = parameters?.ToList() ?? [];
        this.ReturnParameter = returnParameter;
    }

    /// <summary>
    /// Gets the name of the function.
    /// </summary>
    public string FunctionName { get; }

    /// <summary>
    /// Gets the description of the function.
    /// </summary>
    public string Description { get; }

    /// <summary>
    /// Gets the parameters of the function.
    /// </summary>
    public IReadOnlyList<KernelParameterMetadata> Parameters { get; }

    /// <summary>
    /// Gets the return parameter of the function.
    /// </summary>
    public KernelReturnParameterMetadata? ReturnParameter { get; }

    /// <summary>
    /// Gets the name of the function (for compatibility with KernelFunction).
    /// </summary>
    public string Name => this.FunctionName;

    /// <summary>
    /// Gets the plugin name (for compatibility; ONNX functions do not have plugins).
    /// </summary>
    public string? PluginName => null;

    /// <summary>
    /// Gets the JSON schema representation of the function.
    /// </summary>
    public JsonObject ToFunctionDefinition()
    {
        var functionDefinition = new JsonObject
        {
            ["name"] = this.FunctionName,
            ["description"] = this.Description
        };

        if (this.Parameters.Count > 0)
        {
            var properties = new JsonObject();
            var required = new JsonArray();

            foreach (var parameter in this.Parameters)
            {
                var parameterSchema = GetParameterSchema(parameter);
                properties[parameter.Name] = parameterSchema;

                if (parameter.IsRequired)
                {
                    required.Add(parameter.Name);
                }
            }

            functionDefinition["parameters"] = new JsonObject
            {
                ["type"] = "object",
                ["properties"] = properties,
                ["required"] = required
            };
        }

        return functionDefinition;
    }

    /// <summary>
    /// Gets the JSON schema for a parameter.
    /// </summary>
    private static JsonObject GetParameterSchema(KernelParameterMetadata parameter)
    {
        var schema = new JsonObject();

        // Set type based on parameter type
        var type = parameter.ParameterType;
        if (type == typeof(string))
        {
            schema["type"] = "string";
        }
        else if (type == typeof(int) || type == typeof(int?))
        {
            schema["type"] = "integer";
        }
        else if (type == typeof(double) || type == typeof(float) || type == typeof(double?) || type == typeof(float?))
        {
            schema["type"] = "number";
        }
        else if (type == typeof(bool) || type == typeof(bool?))
        {
            schema["type"] = "boolean";
        }
        else if (type.IsArray || (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(List<>)))
        {
            schema["type"] = "array";
        }
        else
        {
            schema["type"] = "object";
        }

        // Add description if available
        if (!string.IsNullOrEmpty(parameter.Description))
        {
            schema["description"] = parameter.Description;
        }

        // Add default value if available
        if (parameter.DefaultValue is not null)
        {
            schema["default"] = JsonSerializer.SerializeToNode(parameter.DefaultValue);
        }

        return schema;
    }

    /// <summary>
    /// Parses a function call from the model's response.
    /// </summary>
    /// <param name="functionName">The name of the function called.</param>
    /// <param name="argumentsJson">The JSON string containing the function arguments.</param>
    /// <returns>A <see cref="FunctionCallContent"/> representing the function call.</returns>
    public static FunctionCallContent ParseFunctionCall(string functionName, string argumentsJson)
    {
        var arguments = new KernelArguments();

        if (!string.IsNullOrEmpty(argumentsJson))
        {
            try
            {
                var jsonDocument = JsonDocument.Parse(argumentsJson);
                foreach (var property in jsonDocument.RootElement.EnumerateObject())
                {
                    // Parse the actual value instead of raw text
                    arguments[property.Name] = property.Value.ValueKind switch
                    {
                        JsonValueKind.String => property.Value.GetString(),
                        JsonValueKind.Number => property.Value.TryGetInt32(out int intValue) ? intValue : property.Value.GetDouble(),
                        JsonValueKind.True => true,
                        JsonValueKind.False => false,
                        JsonValueKind.Null => null,
                        _ => property.Value.GetRawText()
                    };
                }
            }
            catch (JsonException)
            {
                // If parsing fails, treat the entire string as a single argument
                arguments["input"] = argumentsJson;
            }
        }

        return new FunctionCallContent(functionName, arguments: arguments);
    }

    /// <summary>
    /// Creates a function result content from the function result.
    /// </summary>
    /// <param name="functionCallContent">The function call content.</param>
    /// <param name="result">The result of the function call.</param>
    /// <returns>A <see cref="FunctionResultContent"/> representing the function result.</returns>
    public static FunctionResultContent CreateFunctionResult(FunctionCallContent functionCallContent, object? result)
    {
        var resultString = result switch
        {
            string s => s,
            null => "null",
            _ => JsonSerializer.Serialize(result)
        };

        return new FunctionResultContent(functionCallContent, resultString);
    }
}

/// <summary>
/// JSON converter for <see cref="OnnxFunction"/>.
/// </summary>
public sealed class OnnxFunctionJsonConverter : JsonConverter<OnnxFunction>
{
    /// <inheritdoc/>
    public override OnnxFunction? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType != JsonTokenType.StartObject)
        {
            throw new JsonException("Expected start of object.");
        }

        string? functionName = null;
        string? description = null;
        List<KernelParameterMetadata>? parameters = null;

        while (reader.Read())
        {
            if (reader.TokenType == JsonTokenType.EndObject)
            {
                break;
            }

            if (reader.TokenType == JsonTokenType.PropertyName)
            {
                string propertyName = reader.GetString()!;
                reader.Read();

                switch (propertyName)
                {
                    case "name":
                        functionName = reader.GetString();
                        break;
                    case "description":
                        description = reader.GetString();
                        break;
                    case "parameters":
                        // For simplicity, we'll skip parsing parameters in the JSON converter
                        // In a real implementation, you might want to parse these
                        reader.Skip();
                        break;
                }
            }
        }

        if (functionName is null || description is null)
        {
            throw new JsonException("Missing required properties.");
        }

        return new OnnxFunction(functionName, description, parameters);
    }

    /// <inheritdoc/>
    public override void Write(Utf8JsonWriter writer, OnnxFunction value, JsonSerializerOptions options)
    {
        var functionDefinition = value.ToFunctionDefinition();
        functionDefinition.WriteTo(writer);
    }
}

/// <summary>
/// Extension methods for <see cref="KernelFunctionMetadata"/> to convert to <see cref="OnnxFunction"/>.
/// </summary>
public static class OnnxFunctionExtensions
{
    /// <summary>
    /// Converts a <see cref="KernelFunctionMetadata"/> to an <see cref="OnnxFunction"/>.
    /// </summary>
    /// <param name="metadata">The function metadata.</param>
    /// <returns>An <see cref="OnnxFunction"/> representation of the metadata.</returns>
    public static OnnxFunction ToOnnxFunction(this KernelFunctionMetadata metadata)
    {
        return new OnnxFunction(
            functionName: metadata.Name,
            description: metadata.Description,
            parameters: metadata.Parameters,
            returnParameter: metadata.ReturnParameter);
    }
}
