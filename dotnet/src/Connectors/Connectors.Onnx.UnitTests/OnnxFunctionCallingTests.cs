// Copyright (c) Microsoft. All rights reserved.

using System;
using System.ComponentModel;
using System.Threading.Tasks;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.Onnx;
using Xunit;

namespace Microsoft.SemanticKernel.Connectors.Onnx.UnitTests;

/// <summary>
/// Unit tests for ONNX function calling functionality.
/// </summary>
public sealed class OnnxFunctionCallingTests
{
    [Fact]
    public void OnnxToolCallBehavior_EnableKernelFunctions_CreatesCorrectBehavior()
    {
        // Arrange & Act
        var behavior = OnnxToolCallBehavior.EnableKernelFunctions;

        // Assert
        Assert.NotNull(behavior);
        Assert.False(behavior.AutoInvoke);
        Assert.True(behavior.AllowAnyRequestedKernelFunction);
        Assert.Equal(0, behavior.MaximumAutoInvokeAttempts);
    }

    [Fact]
    public void OnnxToolCallBehavior_AutoInvokeKernelFunctions_CreatesCorrectBehavior()
    {
        // Arrange & Act
        var behavior = OnnxToolCallBehavior.AutoInvokeKernelFunctions;

        // Assert
        Assert.NotNull(behavior);
        Assert.True(behavior.AutoInvoke);
        Assert.True(behavior.AllowAnyRequestedKernelFunction);
        Assert.Equal(128, behavior.MaximumAutoInvokeAttempts);
    }

    [Fact]
    public void OnnxToolCallBehavior_EnableFunctions_CreatesCorrectBehavior()
    {
        // Arrange
        var functions = new[]
        {
            new OnnxFunction("TestFunction", "A test function"),
            new OnnxFunction("AnotherFunction", "Another test function")
        };

        // Act
        var behavior = OnnxToolCallBehavior.EnableFunctions(functions, autoInvoke: true);

        // Assert
        Assert.NotNull(behavior);
        Assert.True(behavior.AutoInvoke);
        Assert.Equal(128, behavior.MaximumAutoInvokeAttempts);
    }

    [Fact]
    public void OnnxToolCallBehavior_RequireFunction_CreatesCorrectBehavior()
    {
        // Arrange
        var function = new OnnxFunction("RequiredFunction", "A required function");

        // Act
        var behavior = OnnxToolCallBehavior.RequireFunction(function, autoInvoke: false);

        // Assert
        Assert.NotNull(behavior);
        Assert.False(behavior.AutoInvoke);
        Assert.Equal(0, behavior.MaximumAutoInvokeAttempts);
    }

    [Fact]
    public void OnnxFunction_Constructor_InitializesCorrectly()
    {
        // Arrange
        var functionName = "TestFunction";
        var description = "A test function";

        // Act
        var function = new OnnxFunction(functionName, description);

        // Assert
        Assert.Equal(functionName, function.FunctionName);
        Assert.Equal(description, function.Description);
        Assert.NotNull(function.Parameters);
        Assert.Empty(function.Parameters);
        Assert.Null(function.ReturnParameter);
    }

    [Fact]
    public void OnnxFunction_WithParameters_InitializesCorrectly()
    {
        // Arrange
        var functionName = "TestFunction";
        var description = "A test function";
        var parameters = new[]
        {
            new KernelParameterMetadata("param1")
            {
                Description = "First parameter",
                ParameterType = typeof(string),
                IsRequired = true
            },
            new KernelParameterMetadata("param2")
            {
                Description = "Second parameter",
                ParameterType = typeof(int),
                IsRequired = false
            }
        };

        // Act
        var function = new OnnxFunction(functionName, description, parameters);

        // Assert
        Assert.Equal(functionName, function.FunctionName);
        Assert.Equal(description, function.Description);
        Assert.Equal(2, function.Parameters.Count);
        Assert.Equal("param1", function.Parameters[0].Name);
        Assert.Equal("param2", function.Parameters[1].Name);
    }

    [Fact]
    public void OnnxFunction_ToFunctionDefinition_GeneratesCorrectSchema()
    {
        // Arrange
        var functionName = "TestFunction";
        var description = "A test function";
        var parameters = new[]
        {
            new KernelParameterMetadata("stringParam")
            {
                Description = "A string parameter",
                ParameterType = typeof(string),
                IsRequired = true
            },
            new KernelParameterMetadata("intParam")
            {
                Description = "An integer parameter",
                ParameterType = typeof(int),
                IsRequired = false
            }
        };

        var function = new OnnxFunction(functionName, description, parameters);

        // Act
        var schema = function.ToFunctionDefinition();

        // Assert
        Assert.NotNull(schema);
        Assert.Equal(functionName, schema["name"]?.ToString());
        Assert.Equal(description, schema["description"]?.ToString());
        Assert.NotNull(schema["parameters"]);
        
        var parametersSchema = schema["parameters"];
        Assert.Equal("object", parametersSchema?["type"]?.ToString());
        Assert.NotNull(parametersSchema?["properties"]);
        Assert.NotNull(parametersSchema?["required"]);
    }

    [Fact]
    public void OnnxFunction_ParseFunctionCall_WithValidJson_ParsesCorrectly()
    {
        // Arrange
        var functionName = "TestFunction";
        var argumentsJson = """{"param1": "value1", "param2": 42}""";

        // Act
        var functionCall = OnnxFunction.ParseFunctionCall(functionName, argumentsJson);

        // Assert
        Assert.NotNull(functionCall);
        Assert.Equal(functionName, functionCall.FunctionName);
        Assert.Equal(2, functionCall.Arguments.Count);
        Assert.Contains("param1", functionCall.Arguments);
        Assert.Contains("param2", functionCall.Arguments);
    }

    [Fact]
    public void OnnxFunction_ParseFunctionCall_WithInvalidJson_FallsBackToSingleArgument()
    {
        // Arrange
        var functionName = "TestFunction";
        var invalidJson = "not valid json";

        // Act
        var functionCall = OnnxFunction.ParseFunctionCall(functionName, invalidJson);

        // Assert
        Assert.NotNull(functionCall);
        Assert.Equal(functionName, functionCall.FunctionName);
        Assert.Single(functionCall.Arguments);
        Assert.Contains("input", functionCall.Arguments);
        Assert.Equal(invalidJson, functionCall.Arguments["input"]);
    }

    [Fact]
    public void OnnxFunction_CreateFunctionResult_CreatesCorrectResult()
    {
        // Arrange
        var functionCall = new FunctionCallContent("TestFunction", new KernelArguments());
        var result = "Test result";

        // Act
        var functionResult = OnnxFunction.CreateFunctionResult(functionCall, result);

        // Assert
        Assert.NotNull(functionResult);
        Assert.Equal(functionCall, functionResult.FunctionCall);
        Assert.Equal(result, functionResult.Result);
    }

    [Fact]
    public void OnnxFunctionExtensions_ToOnnxFunction_ConvertsCorrectly()
    {
        // Arrange
        var metadata = new KernelFunctionMetadata("TestFunction")
        {
            Description = "A test function",
            Parameters = new[]
            {
                new KernelParameterMetadata("param1")
                {
                    Description = "First parameter",
                    ParameterType = typeof(string),
                    IsRequired = true
                }
            }
        };

        // Act
        var onnxFunction = metadata.ToOnnxFunction();

        // Assert
        Assert.NotNull(onnxFunction);
        Assert.Equal(metadata.Name, onnxFunction.FunctionName);
        Assert.Equal(metadata.Description, onnxFunction.Description);
        Assert.Equal(metadata.Parameters.Count, onnxFunction.Parameters.Count);
    }

    [Fact]
    public void OnnxRuntimeGenAIPromptExecutionSettings_ToolCallBehavior_SetAndGetCorrectly()
    {
        // Arrange
        var settings = new OnnxRuntimeGenAIPromptExecutionSettings();
        var behavior = OnnxToolCallBehavior.AutoInvokeKernelFunctions;

        // Act
        settings.ToolCallBehavior = behavior;

        // Assert
        Assert.Equal(behavior, settings.ToolCallBehavior);
    }
}

/// <summary>
/// Sample plugin for testing function calling.
/// </summary>
public class TestPlugin
{
    [KernelFunction]
    [Description("Get a test string")]
    public string GetTestString()
    {
        return "Test string";
    }

    [KernelFunction]
    [Description("Add two numbers")]
    public int Add([Description("First number")] int a, [Description("Second number")] int b)
    {
        return a + b;
    }
}