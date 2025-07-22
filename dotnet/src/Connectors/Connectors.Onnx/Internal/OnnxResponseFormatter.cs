// Copyright (c) Microsoft. All rights reserved.

namespace Microsoft.SemanticKernel.Connectors.Onnx.Internal;

/// <summary>
/// Handles formatting of function results for display.
/// </summary>
internal static class OnnxResponseFormatter
{
    /// <summary>
    /// Formats function result for better user experience.
    /// </summary>
    public static string FormatFunctionResult(string functionName, string functionResult)
    {
        // Handle null/empty results (typically from void functions)
        if (string.IsNullOrEmpty(functionResult) || functionResult == "<null>")
        {
            return "Done.";
        }

        // Handle boolean results more naturally
        if (bool.TryParse(functionResult, out bool boolResult))
        {
            return boolResult ? "Yes." : "No.";
        }

        // Return the result as-is for all other cases
        return functionResult;
    }
}