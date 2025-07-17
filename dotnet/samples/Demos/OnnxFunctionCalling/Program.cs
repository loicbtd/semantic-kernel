using System;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.Onnx;
using OnnxFunctionCalling;

var builder = Kernel.CreateBuilder();
var modelId = "llama3.2";
var modelPath = "D:\\VisionCATS_AI_Agent\\Development\\.cache\\models\\microsoft_Phi_4_multimodal_instruct_onnx-gpu_gpu_int4_rtn_block_32";

builder.Services.AddOnnxRuntimeGenAIChatCompletion(modelId, modelPath);

builder.Plugins
    .AddFromType<MyTimePlugin>()
    .AddFromObject(new MyLightPlugin(turnedOn: true))
    .AddFromObject(new MyAlarmPlugin("11"));

var kernel = builder.Build();
var chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();
var settings = new OnnxRuntimeGenAIPromptExecutionSettings
{
    FunctionChoiceBehavior = FunctionChoiceBehavior.Auto()
};

Console.WriteLine("""
                  Ask questions or give instructions to the copilot such as:
                  - Change the alarm to 8
                  - What is the current alarm set?
                  - Is the light on?
                  - Turn the light off please.
                  - Set an alarm for 6:00 am.
                  """);

Console.Write("> ");

string? input = null;
while ((input = Console.ReadLine()) is not null)
{
    Console.WriteLine();

    try
    {
        ChatMessageContent chatResult = await chatCompletionService.GetChatMessageContentAsync(input, settings, kernel);
        Console.Write($"\n>>> Result: {chatResult}\n\n> ");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error: {ex.Message}\n\n> ");
    }
}
