// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.Onnx;
using Microsoft.SemanticKernel.Embeddings;

namespace Microsoft.SemanticKernel;

/// <summary>
/// Provides extension methods for the <see cref="IKernelBuilder"/> class to configure ONNX connectors.
/// </summary>
public static class OnnxKernelBuilderExtensions
{
    /// <summary>
    /// Add OnnxRuntimeGenAI Chat Completion services to the kernel builder.
    /// </summary>
    /// <param name="builder">The kernel builder.</param>
    /// <param name="modelId">Model Id.</param>
    /// <param name="modelPath">The generative AI ONNX model path.</param>
    /// <param name="serviceId">The optional service ID.</param>
    /// <param name="jsonSerializerOptions">The <see cref="JsonSerializerOptions"/> to use for various aspects of serialization, such as function argument deserialization, function result serialization, logging, etc., of the service.</param>
    /// <returns>The updated kernel builder.</returns>
    public static IKernelBuilder AddOnnxRuntimeGenAIChatCompletion(
        this IKernelBuilder builder,
        string modelId,
        string modelPath,
        string? serviceId = null,
        JsonSerializerOptions? jsonSerializerOptions = null)
    {
        builder.Services.AddKeyedSingleton<IChatCompletionService>(serviceId, (serviceProvider, _) =>
            new OnnxRuntimeGenAIChatCompletionService(
                modelId,
                modelPath: modelPath,
                loggerFactory: serviceProvider.GetService<ILoggerFactory>(),
                jsonSerializerOptions));

        return builder;
    }

    /// <summary>
    /// Adds the ONNX GenAI chat completion service with function calling support to the <see cref="IKernelBuilder"/>.
    /// </summary>
    /// <param name="builder">The <see cref="IKernelBuilder"/> instance to augment.</param>
    /// <param name="modelId">The name of the model.</param>
    /// <param name="modelPath">The generative AI ONNX model path for the chat completion service.</param>
    /// <param name="serviceId">A local identifier for the given AI service.</param>
    /// <param name="jsonSerializerOptions">The <see cref="JsonSerializerOptions"/> to use for various aspects of serialization, such as function argument deserialization, function result serialization, logging, etc., of the service.</param>
    /// <param name="providers">The providers to use for the chat completion service.</param>
    /// <returns>The same instance as <paramref name="builder"/>.</returns>
    public static IKernelBuilder AddOnnxRuntimeGenAIFunctionCallingChatCompletion(
        this IKernelBuilder builder,
        string modelId,
        string modelPath,
        string? serviceId = null,
        JsonSerializerOptions? jsonSerializerOptions = null,
        List<string>? providers = null)
    {
        Verify.NotNull(builder);
        Verify.NotNullOrWhiteSpace(modelId);
        Verify.NotNullOrWhiteSpace(modelPath);

        builder.Services.AddKeyedSingleton<IChatCompletionService>(serviceId, (serviceProvider, _) =>
            new OnnxRuntimeGenAIFunctionCallingChatCompletionService(
                modelId,
                modelPath: modelPath,
                loggerFactory: serviceProvider.GetService<ILoggerFactory>(),
                jsonSerializerOptions,
                providers));

        return builder;
    }

    /// <summary>Adds a text embedding generation service using a BERT ONNX model.</summary>
    /// <param name="builder">The <see cref="IKernelBuilder"/> instance to augment.</param>
    /// <param name="onnxModelPath">The path to the ONNX model file.</param>
    /// <param name="vocabPath">The path to the vocab file.</param>
    /// <param name="options">Options for the configuration of the model and service.</param>
    /// <param name="serviceId">A local identifier for the given AI service.</param>
    /// <returns>The same instance as <paramref name="builder"/>.</returns>
    [Obsolete("Use AddBertOnnxEmbeddingGenerator instead")]
#pragma warning disable CA2000 // Dispose objects before losing scope
    public static IKernelBuilder AddBertOnnxTextEmbeddingGeneration(
        this IKernelBuilder builder,
        string onnxModelPath,
        string vocabPath,
        BertOnnxOptions? options = null,
        string? serviceId = null)
    {
        builder.Services.AddKeyedSingleton<ITextEmbeddingGenerationService>(
            serviceId,
            BertOnnxTextEmbeddingGenerationService.Create(onnxModelPath, vocabPath, options));

        return builder;
    }

    /// <summary>Adds a text embedding generation service using a BERT ONNX model.</summary>
    /// <param name="builder">The <see cref="IKernelBuilder"/> instance to augment.</param>
    /// <param name="onnxModelStream">Stream containing the ONNX model. The stream will be read during this call and will not be used after this call's completion.</param>
    /// <param name="vocabStream">Stream containing the vocab file. The stream will be read during this call and will not be used after this call's completion.</param>
    /// <param name="options">Options for the configuration of the model and service.</param>
    /// <param name="serviceId">A local identifier for the given AI service.</param>
    /// <returns>The same instance as <paramref name="builder"/>.</returns>
    [Obsolete("Use AddBertOnnxEmbeddingGenerator instead")]
    public static IKernelBuilder AddBertOnnxTextEmbeddingGeneration(
        this IKernelBuilder builder,
        Stream onnxModelStream,
        Stream vocabStream,
        BertOnnxOptions? options = null,
        string? serviceId = null)
    {
        builder.Services.AddKeyedSingleton<ITextEmbeddingGenerationService>(
            serviceId,
            BertOnnxTextEmbeddingGenerationService.Create(onnxModelStream, vocabStream, options));

        return builder;
    }
#pragma warning restore CA2000 // Dispose objects before losing scope

    /// <summary>Adds a text embedding generation service using a BERT ONNX model.</summary>
    /// <param name="builder">The <see cref="IKernelBuilder"/> instance to augment.</param>
    /// <param name="onnxModelPath">The path to the ONNX model file.</param>
    /// <param name="vocabPath">The path to the vocab file.</param>
    /// <param name="options">Options for the configuration of the model and service.</param>
    /// <param name="serviceId">A local identifier for the given AI service.</param>
    /// <returns>The same instance as <paramref name="builder"/>.</returns>
    public static IKernelBuilder AddBertOnnxEmbeddingGenerator(
        this IKernelBuilder builder,
        string onnxModelPath,
        string vocabPath,
        BertOnnxOptions? options = null,
        string? serviceId = null)
    {
        Verify.NotNull(builder);

        builder.Services.AddBertOnnxEmbeddingGenerator(
            onnxModelPath,
            vocabPath,
            options,
            serviceId);

        return builder;
    }

    /// <summary>Adds a text embedding generation service using a BERT ONNX model.</summary>
    /// <param name="builder">The <see cref="IKernelBuilder"/> instance to augment.</param>
    /// <param name="onnxModelStream">Stream containing the ONNX model. The stream will be read during this call and will not be used after this call's completion.</param>
    /// <param name="vocabStream">Stream containing the vocab file. The stream will be read during this call and will not be used after this call's completion.</param>
    /// <param name="options">Options for the configuration of the model and service.</param>
    /// <param name="serviceId">A local identifier for the given AI service.</param>
    /// <returns>The same instance as <paramref name="builder"/>.</returns>
    public static IKernelBuilder AddBertOnnxEmbeddingGenerator(
        this IKernelBuilder builder,
        Stream onnxModelStream,
        Stream vocabStream,
        BertOnnxOptions? options = null,
        string? serviceId = null)
    {
        Verify.NotNull(builder);

        builder.Services.AddBertOnnxEmbeddingGenerator(
            onnxModelStream,
            vocabStream,
            options,
            serviceId);

        return builder;
    }
}
