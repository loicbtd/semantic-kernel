// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text.Json;
using Microsoft.SemanticKernel.ChatCompletion;

namespace Microsoft.SemanticKernel.Connectors.Onnx;

/// <summary>Represents a behavior for ONNX tool calls.</summary>
public abstract class OnnxToolCallBehavior
{
    /// <summary>
    /// The default maximum number of tool-call auto-invokes that can be made in a single request.
    /// </summary>
    /// <remarks>
    /// After this number of iterations as part of a single user request is reached, auto-invocation
    /// will be disabled (e.g. <see cref="AutoInvokeKernelFunctions"/> will behave like <see cref="EnableKernelFunctions"/>).
    /// This is a safeguard against possible runaway execution if the model routinely re-requests
    /// the same function over and over. It is currently hardcoded, but in the future it could
    /// be made configurable by the developer.
    /// </remarks>
    private const int DefaultMaximumAutoInvokeAttempts = 128;

    /// <summary>
    /// Gets an instance that will provide all of the <see cref="Kernel"/>'s plugins' function information.
    /// Function call requests from the model will be propagated back to the caller.
    /// </summary>
    /// <remarks>
    /// If no <see cref="Kernel"/> is available, no function information will be provided to the model.
    /// </remarks>
    public static OnnxToolCallBehavior EnableKernelFunctions { get; } = new KernelFunctions(autoInvoke: false);

    /// <summary>
    /// Gets an instance that will both provide all of the <see cref="Kernel"/>'s plugins' function information
    /// to the model and attempt to automatically handle any function call requests.
    /// </summary>
    /// <remarks>
    /// When successful, tool call requests from the model become an implementation detail, with the service
    /// handling invoking any requested functions and supplying the results back to the model.
    /// If no <see cref="Kernel"/> is available, no function information will be provided to the model.
    /// </remarks>
    public static OnnxToolCallBehavior AutoInvokeKernelFunctions { get; } = new KernelFunctions(autoInvoke: true);

    /// <summary>Gets an instance that will provide the specified list of functions to the model.</summary>
    /// <param name="functions">The functions that should be made available to the model.</param>
    /// <param name="autoInvoke">true to attempt to automatically handle function call requests; otherwise, false.</param>
    /// <returns>
    /// The <see cref="OnnxToolCallBehavior"/> that may be set into <see cref="OnnxRuntimeGenAIPromptExecutionSettings.ToolCallBehavior"/>
    /// to indicate that the specified functions should be made available to the model.
    /// </returns>
    public static OnnxToolCallBehavior EnableFunctions(IEnumerable<OnnxFunction> functions, bool autoInvoke = false)
    {
        Verify.NotNull(functions);
        return new EnabledFunctions(functions, autoInvoke);
    }

    /// <summary>Gets an instance that will request the model to use the specified function.</summary>
    /// <param name="function">The function the model should request to use.</param>
    /// <param name="autoInvoke">true to attempt to automatically handle function call requests; otherwise, false.</param>
    /// <returns>
    /// The <see cref="OnnxToolCallBehavior"/> that may be set into <see cref="OnnxRuntimeGenAIPromptExecutionSettings.ToolCallBehavior"/>
    /// to indicate that the specified function should be requested by the model.
    /// </returns>
    public static OnnxToolCallBehavior RequireFunction(OnnxFunction function, bool autoInvoke = false)
    {
        Verify.NotNull(function);
        return new RequiredFunction(function, autoInvoke);
    }

    /// <summary>Initializes the instance; prevents external instantiation.</summary>
    private OnnxToolCallBehavior(bool autoInvoke)
    {
        this.MaximumAutoInvokeAttempts = autoInvoke ? DefaultMaximumAutoInvokeAttempts : 0;
    }

    /// <summary>
    /// Options to control tool call result serialization behavior.
    /// </summary>
    [Obsolete("This property is deprecated in favor of Kernel.SerializerOptions that will be introduced in one of the following releases.")]
    [ExcludeFromCodeCoverage]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public virtual JsonSerializerOptions? ToolCallResultSerializerOptions { get; set; }

    /// <summary>Gets the maximum number of tool-call auto-invokes that can be made in a single request.</summary>
    public int MaximumAutoInvokeAttempts { get; private init; }

    /// <summary>Gets whether to automatically invoke functions.</summary>
    public bool AutoInvoke => this.MaximumAutoInvokeAttempts > 0;

    /// <summary>Gets whether to allow invocation of kernel functions that weren't advertised to the model.</summary>
    /// <remarks>
    /// When this property is set to <see langword="true"/>, the model can request to invoke any function
    /// available in the <see cref="Kernel"/>, regardless of whether it was advertised to the model or not.
    /// When this property is set to <see langword="false"/>, the model can only request to invoke functions
    /// that were advertised to the model.
    /// </remarks>
    public virtual bool AllowAnyRequestedKernelFunction { get; internal set; }

    /// <summary>Gets the options to use when processing function calls.</summary>
    public abstract FunctionChoiceBehaviorOptions? Options { get; }

    /// <summary>Configures the request with any tool information this <see cref="OnnxToolCallBehavior"/> provides.</summary>
    /// <param name="kernel">The <see cref="Kernel"/> to use for function calling.</param>
    /// <param name="chatHistory">The chat history to use for function calling.</param>
    /// <param name="requestIndex">The request index.</param>
    /// <returns>The tools configuration for the request.</returns>
    internal abstract OnnxToolCallingConfig ConfigureRequest(Kernel? kernel, ChatHistory chatHistory, int requestIndex);

    /// <summary>
    /// Clones the current instance.
    /// </summary>
    /// <returns>A new instance that is a copy of the current instance.</returns>
    internal OnnxToolCallBehavior Clone()
    {
        return (OnnxToolCallBehavior)this.MemberwiseClone();
    }

    /// <summary>
    /// Represents a <see cref="OnnxToolCallBehavior"/> that will provide to the model all available functions from a
    /// <see cref="Kernel"/>.
    /// </summary>
    internal sealed class KernelFunctions : OnnxToolCallBehavior
    {
        /// <summary>Initializes the instance.</summary>
        /// <param name="autoInvoke">true to attempt to automatically handle function call requests; otherwise, false.</param>
        public KernelFunctions(bool autoInvoke) : base(autoInvoke)
        {
            this.AllowAnyRequestedKernelFunction = true;
        }

        /// <inheritdoc/>
        public override FunctionChoiceBehaviorOptions? Options { get; } = new();

        /// <inheritdoc/>
        internal override OnnxToolCallingConfig ConfigureRequest(Kernel? kernel, ChatHistory chatHistory, int requestIndex)
        {
            return new OnnxToolCallingConfig(
                Tools: GetKernelFunctions(kernel),
                AutoInvoke: this.AutoInvoke,
                AllowAnyRequestedKernelFunction: this.AllowAnyRequestedKernelFunction,
                Options: this.Options);
        }

        /// <summary>Gets the functions from the kernel.</summary>
        private static IList<OnnxFunction>? GetKernelFunctions(Kernel? kernel)
        {
            if (kernel is null)
            {
                return null;
            }

            return kernel.Plugins.GetFunctionsMetadata().Select(f => f.ToOnnxFunction()).ToList();
        }
    }

    /// <summary>
    /// Represents a <see cref="OnnxToolCallBehavior"/> that provides a specified list of functions to the model.
    /// </summary>
    internal sealed class EnabledFunctions(IEnumerable<OnnxFunction> functions, bool autoInvoke) : OnnxToolCallBehavior(autoInvoke)
    {
        /// <inheritdoc/>
        public override FunctionChoiceBehaviorOptions? Options { get; } = new();

        /// <inheritdoc/>
        internal override OnnxToolCallingConfig ConfigureRequest(Kernel? kernel, ChatHistory chatHistory, int requestIndex)
        {
            return new OnnxToolCallingConfig(
                Tools: functions.ToList(),
                AutoInvoke: this.AutoInvoke,
                AllowAnyRequestedKernelFunction: this.AllowAnyRequestedKernelFunction,
                Options: this.Options);
        }
    }

    /// <summary>
    /// Represents a <see cref="OnnxToolCallBehavior"/> that requests a specific function to be invoked.
    /// </summary>
    internal sealed class RequiredFunction(OnnxFunction function, bool autoInvoke) : OnnxToolCallBehavior(autoInvoke)
    {
        /// <inheritdoc/>
        public override FunctionChoiceBehaviorOptions? Options { get; } = new();

        /// <inheritdoc/>
        internal override OnnxToolCallingConfig ConfigureRequest(Kernel? kernel, ChatHistory chatHistory, int requestIndex)
        {
            return new OnnxToolCallingConfig(
                Tools: [function],
                AutoInvoke: this.AutoInvoke,
                AllowAnyRequestedKernelFunction: this.AllowAnyRequestedKernelFunction,
                Options: this.Options);
        }
    }
}

/// <summary>
/// Configuration for ONNX tool calling.
/// </summary>
/// <param name="Tools">The list of tools available to the model.</param>
/// <param name="AutoInvoke">Whether to automatically invoke functions.</param>
/// <param name="AllowAnyRequestedKernelFunction">Whether to allow invocation of kernel functions that weren't advertised to the model.</param>
/// <param name="Options">The options to use when processing function calls.</param>
internal record OnnxToolCallingConfig(
    IList<OnnxFunction>? Tools,
    bool AutoInvoke,
    bool AllowAnyRequestedKernelFunction,
    FunctionChoiceBehaviorOptions? Options);