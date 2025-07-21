using System.ComponentModel;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.Onnx;

var builder = Kernel.CreateBuilder();

// Use the function calling version instead of the basic one
builder.AddOnnxRuntimeGenAIFunctionCallingChatCompletion(
    "phi-4-multimodal-instruct",
    "D:\\VisionCATS_AI_Agent\\Development\\.cache\\models\\microsoft_Phi_4_multimodal_instruct_onnx-gpu_gpu_int4_rtn_block_32",
    providers: ["cuda"]);

builder.Plugins.AddFromType<DefaultPlugin>();

var kernel = builder.Build();

// Use chat completion service
var history = new ChatHistory();
history.AddSystemMessage("""
                         You are a helpful assistant.
                         You are not restricted to using the provided plugins,
                         and you can use information from your training.
                         Please explain your reasoning with the response.
                         """);

var chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();

// Enable function calling with auto-invocation
var executionSettings = new OnnxRuntimeGenAIPromptExecutionSettings
{
    ToolCallBehavior = OnnxToolCallBehavior.AutoInvokeKernelFunctions
};

while (true)
{
    Console.Write("User > ");
    var userMessage = Console.ReadLine();
    if (userMessage == "exit" || userMessage == "quit")
    {
        break;
    }

    if (string.IsNullOrEmpty(userMessage))
    {
        continue;
    }

    history.AddUserMessage(userMessage);

    try
    {
        // Use the non-streaming API for function calling
        var results = await chatCompletionService.GetChatMessageContentsAsync(history, executionSettings, kernel);

        var fullMessage = "";
        foreach (var result in results)
        {
            Console.Write("Assistant > ");
            Console.WriteLine(result.Content);
            fullMessage = result.Content ?? "";
        }

        history.AddAssistantMessage(fullMessage);
    }
    catch (Exception e)
    {
        Console.WriteLine($"Error: {e.Message}");
    }
}

// No explicit dispose needed; kernel manages service lifetimes

public class DefaultPlugin
{
    [KernelFunction]
    [Description("get the current date and time")]
    public static DateTime Time()
    {
        return DateTime.Now;
    }

    [KernelFunction]
    [Description("get the current day of the week")]
    public static DayOfWeek DayOfWeek()
    {
        return DateTime.UtcNow.DayOfWeek;
    }
}

// public class LoggingFilter(Serilog.ILogger logger) : IFunctionInvocationFilter
// {
//     public async Task OnFunctionInvocationAsync(FunctionInvocationContext context, Func<FunctionInvocationContext, Task> next)
//     {
//         logger.Information("FunctionInvoking - {PluginName}.{FunctionName}", context.Function.PluginName, context.Function.Name);
//
//         await next(context);
//
//         logger.Information("FunctionInvoked - {PluginName}.{FunctionName}", context.Function.PluginName, context.Function.Name);
//     }
// }
