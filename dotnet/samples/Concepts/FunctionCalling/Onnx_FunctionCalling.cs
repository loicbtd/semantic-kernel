// Copyright (c) Microsoft. All rights reserved.

using System;
using System.ComponentModel;
using System.Threading.Tasks;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.Onnx;

namespace FunctionCalling;

/// <summary>
/// This example demonstrates how to use function calling with the ONNX connector.
/// </summary>
public class Onnx_FunctionCalling(ITestOutputHelper output) : BaseTest(output)
{
    /// <summary>
    /// Shows how to use function calling with auto-invocation enabled.
    /// </summary>
    [Fact]
    public async Task FunctionCallingAutoInvokeAsync()
    {
        string modelPath = "path/to/your/onnx/model";
        string modelId = "phi-3-mini-4k-instruct";

        // Create a kernel with the ONNX function calling chat completion service
        var builder = Kernel.CreateBuilder()
            .AddOnnxRuntimeGenAIFunctionCallingChatCompletion(modelId, modelPath);
        
        // Add a simple plugin with a few functions
        builder.Plugins.AddFromType<TimePlugin>();
        builder.Plugins.AddFromType<MathPlugin>();
        
        Kernel kernel = builder.Build();

        // Enable auto-invocation of functions
        OnnxRuntimeGenAIPromptExecutionSettings settings = new()
        {
            ToolCallBehavior = OnnxToolCallBehavior.AutoInvokeKernelFunctions
        };

        // Test function calling
        string prompt = "What time is it? Also, what's 15 + 27?";
        var result = await kernel.InvokePromptAsync(prompt, new(settings));

        Console.WriteLine($"Result: {result}");
    }

    /// <summary>
    /// Shows how to use function calling with manual invocation.
    /// </summary>
    [Fact]
    public async Task FunctionCallingManualInvokeAsync()
    {
        string modelPath = "path/to/your/onnx/model";
        string modelId = "phi-3-mini-4k-instruct";

        // Create a kernel with the ONNX function calling chat completion service
        var builder = Kernel.CreateBuilder()
            .AddOnnxRuntimeGenAIFunctionCallingChatCompletion(modelId, modelPath);
        
        // Add a simple plugin with a few functions
        builder.Plugins.AddFromType<TimePlugin>();
        builder.Plugins.AddFromType<MathPlugin>();
        
        Kernel kernel = builder.Build();

        // Enable function calling without auto-invocation
        OnnxRuntimeGenAIPromptExecutionSettings settings = new()
        {
            ToolCallBehavior = OnnxToolCallBehavior.EnableKernelFunctions
        };

        // Test function calling
        string prompt = "What time is it?";
        var result = await kernel.InvokePromptAsync(prompt, new(settings));

        Console.WriteLine($"Result: {result}");
    }

    /// <summary>
    /// Shows how to use function calling with specific functions.
    /// </summary>
    [Fact]
    public async Task FunctionCallingSpecificFunctionsAsync()
    {
        string modelPath = "path/to/your/onnx/model";
        string modelId = "phi-3-mini-4k-instruct";

        // Create a kernel with the ONNX function calling chat completion service
        var builder = Kernel.CreateBuilder()
            .AddOnnxRuntimeGenAIFunctionCallingChatCompletion(modelId, modelPath);
        
        // Add a simple plugin with a few functions
        builder.Plugins.AddFromType<MathPlugin>();
        
        Kernel kernel = builder.Build();

        // Get specific functions
        var mathPlugin = kernel.Plugins["MathPlugin"];
        var addFunction = mathPlugin["Add"];
        var multiplyFunction = mathPlugin["Multiply"];

        // Convert to ONNX functions
        var onnxFunctions = new[]
        {
            addFunction.Metadata.ToOnnxFunction(),
            multiplyFunction.Metadata.ToOnnxFunction()
        };

        // Enable specific functions with auto-invocation
        OnnxRuntimeGenAIPromptExecutionSettings settings = new()
        {
            ToolCallBehavior = OnnxToolCallBehavior.EnableFunctions(onnxFunctions, autoInvoke: true)
        };

        // Test function calling
        string prompt = "What's 15 + 27 and then multiply that result by 3?";
        var result = await kernel.InvokePromptAsync(prompt, new(settings));

        Console.WriteLine($"Result: {result}");
    }
}

/// <summary>
/// A plugin that provides time-related functions.
/// </summary>
public class TimePlugin
{
    [KernelFunction]
    [Description("Get the current time in UTC")]
    public DateTime GetCurrentUtcTime()
    {
        return DateTime.UtcNow;
    }

    [KernelFunction]
    [Description("Get the current time in a specific timezone")]
    public DateTime GetTimeInTimezone([Description("The timezone ID (e.g., 'America/New_York')")] string timezoneId)
    {
        var timeZone = TimeZoneInfo.FindSystemTimeZoneById(timezoneId);
        return TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow, timeZone);
    }
}

/// <summary>
/// A plugin that provides math-related functions.
/// </summary>
public class MathPlugin
{
    [KernelFunction]
    [Description("Add two numbers")]
    public double Add([Description("The first number")] double a, [Description("The second number")] double b)
    {
        return a + b;
    }

    [KernelFunction]
    [Description("Multiply two numbers")]
    public double Multiply([Description("The first number")] double a, [Description("The second number")] double b)
    {
        return a * b;
    }

    [KernelFunction]
    [Description("Calculate the square root of a number")]
    public double SquareRoot([Description("The number to calculate square root of")] double number)
    {
        return Math.Sqrt(number);
    }
}