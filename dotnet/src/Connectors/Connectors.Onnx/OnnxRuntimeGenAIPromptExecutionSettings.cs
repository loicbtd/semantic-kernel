// Copyright (c) Microsoft. All rights reserved.

using System.Diagnostics.CodeAnalysis;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;
using Microsoft.SemanticKernel.Text;

namespace Microsoft.SemanticKernel.Connectors.Onnx;

/// <summary>
/// OnnxRuntimeGenAI Execution Settings.
/// </summary>
[JsonNumberHandling(JsonNumberHandling.AllowReadingFromString)]
public sealed class OnnxRuntimeGenAIPromptExecutionSettings : PromptExecutionSettings
{
    /// <summary>
    /// Convert PromptExecutionSettings to OnnxRuntimeGenAIPromptExecutionSettings
    /// </summary>
    /// <param name="executionSettings">The <see cref="PromptExecutionSettings"/> to convert to <see cref="OnnxRuntimeGenAIPromptExecutionSettings"/>.</param>
    /// <returns>Returns the <see cref="OnnxRuntimeGenAIPromptExecutionSettings"/> object.</returns>
    [RequiresUnreferencedCode("This method uses reflection to serialize and deserialize the execution settings, making it incompatible with AOT scenarios.")]
    [RequiresDynamicCode("This method uses reflection to serialize and deserialize the execution settings, making it incompatible with AOT scenarios.")]
    public static OnnxRuntimeGenAIPromptExecutionSettings FromExecutionSettings(PromptExecutionSettings? executionSettings)
    {
        if (executionSettings is null)
        {
            return new OnnxRuntimeGenAIPromptExecutionSettings();
        }

        if (executionSettings is OnnxRuntimeGenAIPromptExecutionSettings settings)
        {
            return settings;
        }

        var json = JsonSerializer.Serialize(executionSettings, executionSettings.GetType());

        return JsonSerializer.Deserialize<OnnxRuntimeGenAIPromptExecutionSettings>(json, JsonOptionsCache.ReadPermissive)!;
    }

    /// <summary>
    /// Convert PromptExecutionSettings to OnnxRuntimeGenAIPromptExecutionSettings
    /// </summary>
    /// <param name="executionSettings">The <see cref="PromptExecutionSettings"/> to convert to <see cref="OnnxRuntimeGenAIPromptExecutionSettings"/>.</param>
    /// <param name="jsonSerializerOptions">The <see cref="JsonSerializerOptions"/> to use for serialization of <see cref="PromptExecutionSettings"/> and deserialize them to <see cref="OnnxRuntimeGenAIPromptExecutionSettings"/>.</param>
    /// <returns>Returns the <see cref="OnnxRuntimeGenAIPromptExecutionSettings"/> object.</returns>
    public static OnnxRuntimeGenAIPromptExecutionSettings FromExecutionSettings(PromptExecutionSettings? executionSettings, JsonSerializerOptions jsonSerializerOptions)
    {
        if (executionSettings is null)
        {
            return new OnnxRuntimeGenAIPromptExecutionSettings();
        }

        if (executionSettings is OnnxRuntimeGenAIPromptExecutionSettings settings)
        {
            return settings;
        }

        JsonTypeInfo typeInfo = jsonSerializerOptions.GetTypeInfo(executionSettings!.GetType());

        var json = JsonSerializer.Serialize(executionSettings, typeInfo);

        return JsonSerializer.Deserialize<OnnxRuntimeGenAIPromptExecutionSettings>(json, OnnxRuntimeGenAIPromptExecutionSettingsJsonSerializerContext.ReadPermissive.OnnxRuntimeGenAIPromptExecutionSettings)!;
    }

    /// <summary>
    /// Top k tokens to sample from
    /// </summary>
    [JsonPropertyName("top_k")]
    public int? TopK { get; set; }

    /// <summary>
    /// Top p probability to sample with
    /// </summary>
    [JsonPropertyName("top_p")]
    public float? TopP { get; set; }

    /// <summary>
    /// Temperature to sample with
    /// </summary>
    [JsonPropertyName("temperature")]
    public float? Temperature { get; set; }

    /// <summary>
    /// Repetition penalty to sample with
    /// </summary>
    [JsonPropertyName("repetition_penalty")]
    public float? RepetitionPenalty { get; set; }

    /// <summary>
    /// The past/present kv tensors are shared and allocated once to max_length (cuda only)
    /// </summary>
    [JsonPropertyName("past_present_share_buffer")]
    [JsonConverter(typeof(OptionalBoolJsonConverter))]
    public bool? PastPresentShareBuffer { get; set; }

    /// <summary>
    /// The number of independently computed returned sequences for each element in the batch
    /// </summary>
    [JsonPropertyName("num_return_sequences")]
    public int? NumReturnSequences { get; set; }

    /// <summary>
    /// The number of beams used during beam_search
    /// </summary>
    [JsonPropertyName("num_beams")]
    public int? NumBeams { get; set; }

    /// <summary>
    /// No repeated ngram in generated summaries
    /// </summary>
    [JsonPropertyName("no_repeat_ngram_size")]
    public int? NoRepeatNgramSize { get; set; }

    /// <summary>
    /// Min number of tokens to generate including the prompt
    /// </summary>
    [JsonPropertyName("min_tokens")]
    public int? MinTokens { get; set; }

    /// <summary>
    /// Max number of tokens to generate including the prompt
    /// </summary>
    [JsonPropertyName("max_tokens")]
    public int? MaxTokens { get; set; }

    /// <summary>
    /// Length penalty of generated summaries
    /// </summary>
    [JsonPropertyName("length_penalty")]
    public float? LengthPenalty { get; set; }

    /// <summary>
    /// Indicating by which amount to penalize common words between beam group
    /// </summary>
    [JsonPropertyName("diversity_penalty")]
    public float? DiversityPenalty { get; set; }

    /// <summary>
    /// Allows the generation to stop early if all beam candidates reach the end token
    /// </summary>
    [JsonPropertyName("early_stopping")]
    [JsonConverter(typeof(OptionalBoolJsonConverter))]
    public bool? EarlyStopping { get; set; }

    /// <summary>
    /// Do random sampling
    /// </summary>
    [JsonPropertyName("do_sample")]
    [JsonConverter(typeof(OptionalBoolJsonConverter))]
    public bool? DoSample { get; set; }

    /// <summary>
    /// The tool call behavior to use for the ONNX model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The ONNX model can have its behavior configured to use tools. This can be:
    /// </para>
    /// <list type="bullet">
    /// <item>To disable all tool calling, set the property to null (the default).</item>
    /// <item>
    /// To allow the model to request one of any number of functions, set the property to an
    /// instance returned from <see cref="OnnxToolCallBehavior.EnableFunctions"/>, called with
    /// a list of the functions available.
    /// </item>
    /// <item>
    /// To allow the model to request one of any of the functions in the supplied <see cref="Kernel"/>,
    /// set the property to <see cref="OnnxToolCallBehavior.EnableKernelFunctions"/> if the client should simply
    /// send the information about the functions and not handle the response in any special manner, or
    /// <see cref="OnnxToolCallBehavior.AutoInvokeKernelFunctions"/> if the client should attempt to automatically
    /// invoke the function and send the result back to the service.
    /// </item>
    /// </list>
    /// For all options where an instance is provided, auto-invoke behavior may be selected. If the service
    /// sends a request for a function call, if auto-invoke has been requested, the client will attempt to
    /// resolve that function from the functions available in the <see cref="Kernel"/>, and if found, rather
    /// than returning the response back to the caller, it will handle the request automatically, invoking
    /// the function, and sending back the result. The intermediate messages will be retained in the
    /// <see cref="ChatHistory"/> if an instance was provided.
    /// </remarks>
    [JsonIgnore]
    public OnnxToolCallBehavior? ToolCallBehavior { get; set; }
}
