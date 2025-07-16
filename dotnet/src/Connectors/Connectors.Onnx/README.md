# ONNX Connector for Semantic Kernel

This connector provides support for using ONNX models with Semantic Kernel, including function calling capabilities.

## Features

### Basic Chat Completion
- Basic chat completion using ONNX Runtime GenAI
- Streaming and non-streaming responses
- Customizable execution settings

### Function Calling (New)
- **Auto-invocation**: Automatically invoke functions when the model requests them
- **Manual invocation**: Get function call requests and handle them manually
- **Specific function sets**: Configure which functions are available to the model
- **Kernel function integration**: Seamlessly integrate with Semantic Kernel plugins

## Usage

### Basic Setup

```csharp
var builder = Kernel.CreateBuilder()
    .AddOnnxRuntimeGenAIChatCompletion("phi-3-mini-4k-instruct", "path/to/model");

Kernel kernel = builder.Build();
```

### Function Calling Setup

```csharp
var builder = Kernel.CreateBuilder()
    .AddOnnxRuntimeGenAIFunctionCallingChatCompletion("phi-3-mini-4k-instruct", "path/to/model");

// Add plugins
builder.Plugins.AddFromType<TimePlugin>();
builder.Plugins.AddFromType<MathPlugin>();

Kernel kernel = builder.Build();
```

### Auto-Invocation

```csharp
OnnxRuntimeGenAIPromptExecutionSettings settings = new()
{
    ToolCallBehavior = OnnxToolCallBehavior.AutoInvokeKernelFunctions
};

var result = await kernel.InvokePromptAsync("What time is it?", new(settings));
```

### Manual Invocation

```csharp
OnnxRuntimeGenAIPromptExecutionSettings settings = new()
{
    ToolCallBehavior = OnnxToolCallBehavior.EnableKernelFunctions
};

var result = await kernel.InvokePromptAsync("What time is it?", new(settings));
// Handle function calls manually from the result
```

### Specific Functions

```csharp
var onnxFunctions = new[]
{
    myFunction.Metadata.ToOnnxFunction(),
    anotherFunction.Metadata.ToOnnxFunction()
};

OnnxRuntimeGenAIPromptExecutionSettings settings = new()
{
    ToolCallBehavior = OnnxToolCallBehavior.EnableFunctions(onnxFunctions, autoInvoke: true)
};

var result = await kernel.InvokePromptAsync("Use my specific functions", new(settings));
```

## Function Calling Implementation

The ONNX connector implements function calling by:

1. **Function Registration**: Functions are registered with their metadata and JSON schema
2. **Prompt Enhancement**: Function definitions are added to the system prompt
3. **Response Parsing**: The model's response is parsed for function call requests
4. **Function Execution**: Functions are executed and results are fed back to the model

### Enhanced Parsing

The connector now supports multiple parsing strategies:

- **JSON Code Blocks**: Extracts function calls from ```json ... ``` blocks
- **Inline Code**: Extracts function calls from `{...}` inline code blocks
- **Raw JSON**: Parses function calls from direct JSON responses
- **Regex Fallback**: Uses regex patterns for non-standard formats

### Auto-Invoke Fallback

When the standard function calling pipeline fails (common with some ONNX models like Phi-4), the connector automatically falls back to manual function invocation:

1. **Detection**: Identifies function calls in the model's response
2. **Validation**: Verifies the function exists and is allowed
3. **Execution**: Manually invokes the function through the kernel
4. **Result**: Returns the function result instead of the JSON call

This ensures compatibility with models that generate function calls but don't work well with the standard auto-invoke pipeline.

### Function Call Format

The model is instructed to make function calls in the following JSON format:

```json
{
  "function_call": {
    "name": "function_name",
    "arguments": {
      "param1": "value1",
      "param2": "value2"
    }
  }
}
```

### Function Definition Schema

Functions are provided to the model using JSON schema format:

```json
{
  "name": "get_current_time",
  "description": "Get the current time in UTC",
  "parameters": {
    "type": "object",
    "properties": {
      "timezone": {
        "type": "string",
        "description": "The timezone ID"
      }
    },
    "required": ["timezone"]
  }
}
```

## Classes

### OnnxToolCallBehavior
- `EnableKernelFunctions`: Enable all kernel functions without auto-invocation
- `AutoInvokeKernelFunctions`: Enable all kernel functions with auto-invocation
- `EnableFunctions(functions, autoInvoke)`: Enable specific functions
- `RequireFunction(function, autoInvoke)`: Require a specific function

### OnnxFunction
- Represents a function that can be called by the ONNX model
- Provides JSON schema generation for function definitions
- Handles function call parsing and result formatting

### OnnxRuntimeGenAIFunctionCallingChatCompletionService
- Extended chat completion service with function calling support
- Integrates with Semantic Kernel's function calling infrastructure
- Handles both streaming and non-streaming scenarios

## Model Requirements

For function calling to work effectively, the ONNX model should:
- Support instruction following
- Be able to generate structured JSON output
- Have been fine-tuned or trained to understand function calling patterns

### Model Compatibility

**Models with Full Auto-Invoke Support:**
- Llama 2/3 Chat variants
- Mistral variants
- Some Phi-3 variants

**Models with Fallback Support (Manual Invocation):**
- Phi-4 Multimodal Instruct ONNX
- Other models that generate function calls but don't work with standard auto-invoke

The connector automatically detects which approach works best for each model and uses the appropriate method.

### Troubleshooting

If function calling isn't working:

1. **Check Logs**: Enable debug logging to see parsing attempts
2. **Verify Model**: Ensure the model supports instruction following
3. **Test Parsing**: Try different function call formats
4. **Fallback**: The manual fallback should work even if auto-invoke fails

## Configuration

### Execution Settings

```csharp
OnnxRuntimeGenAIPromptExecutionSettings settings = new()
{
    ToolCallBehavior = OnnxToolCallBehavior.AutoInvokeKernelFunctions,
    Temperature = 0.7f,
    TopP = 0.9f,
    MaxTokens = 1000,
    // ... other ONNX-specific settings
};
```

### Function Call Options

```csharp
var options = new FunctionChoiceBehaviorOptions
{
    AllowConcurrentInvocation = true,
    AllowAnyRequestedKernelFunction = false
};
```

## Error Handling

The connector handles various error scenarios:
- JSON parsing errors when interpreting function calls
- Function not found errors
- Function execution errors
- Model timeout or generation errors

## Performance Considerations

- Function definitions are included in the system prompt, which increases token usage
- Auto-invocation requires multiple model calls for complex function chains
- Consider using streaming for better user experience with function calls

## Examples

See the `Onnx_FunctionCalling.cs` sample for comprehensive examples of all function calling scenarios.