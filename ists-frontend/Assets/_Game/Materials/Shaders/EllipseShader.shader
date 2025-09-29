Shader "Talia/EllipseShader"
{
    Properties
    {
        _Mode ("Mode", Float) = 1 // 0 is unclustered, 1 is clustered
        _Timestep ("Timestep", Integer) = 0
        _SelectedNodeID ("SelectedNodeID", Integer) = -1
        _UniversalScaleFactor ("UniversalScaleFactor", Float) = 1
        _CNodesCount ("CNodesCount", Integer) = 0
    }
    SubShader
    {
        Pass
        {
            Tags
            {
                "RenderType"="Opaque"
                "RenderPipeline" = "UniversalRenderPipeline"
            }

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            StructuredBuffer<float2> position_buffer_1; // Positions for the circles
            StructuredBuffer<float> scale_buffer;     // Scale factors buffer for X and Y
            StructuredBuffer<float> clusterActivationFractionBuffer;
            StructuredBuffer<uint> nodeActivationBuffer;
            StructuredBuffer<uint> blacklist_buffer;

            float _Mode;
            uint _Timestep;
            uint _SelectedNodeID;
            uint _CNodesCount;
            float _UniversalScaleFactor;
        
            struct attributes
            {
                float4 vertex : POSITION; // Local vertex position
                float2 uv : TEXCOORD0;   // Texture coordinates
            };

            struct varyings
            {
                float4 vertex : SV_POSITION; // Vertex position
                float4 color : COLOR;       // Vertex color (RGBA)
            };

            varyings vert(attributes v, const uint instance_id : SV_InstanceID)
            {
                varyings o;

                if ((blacklist_buffer[instance_id] == 1) || (_Mode > 0.1f && instance_id >= _CNodesCount) || (_Mode <= 0.1f && instance_id < _CNodesCount))
                {
                    o.vertex = float4(0.0f,0.0f,0.0f,0.0f);
                    o.color = float4(0.0f,0.0f,0.0f,0.0f);
                    return o;
                }

                float2 position = position_buffer_1[instance_id]; // Use 2D position
                float scale = _UniversalScaleFactor;  
                float4 instanceColor = float4(1.0f, 0.0f, 0.0f, 1.0f); // Retrieve color from buffer

                // If node 'instance_id' belongs to the clustered graph
                if (instance_id < _CNodesCount)
                {
                    // Use scale proportional to node size...
                    scale *= scale_buffer[instance_id];
                    // Use fraction from the other buffer to tint
                    instanceColor.g = clusterActivationFractionBuffer[instance_id];
                }
                else if (nodeActivationBuffer[instance_id] <= _Timestep) // If node is unclustered and meant to be active
                {
                    instanceColor.g = 1.0f;
                }

                if (instance_id == _SelectedNodeID) {
                    instanceColor.r = 1.0f;
                    instanceColor.g= 1.0f;
                    instanceColor.b = 1.0f;
                }

                // Apply scale
                float2 scaledVertex = float2(v.vertex.x * scale, v.vertex.y * scale);

                // Apply translation
                float2 world_pos = position + scaledVertex;

                o.vertex = mul(UNITY_MATRIX_VP, float4(world_pos, 0.0, 1.0)); // Transform to clip space
                o.color = instanceColor; // Use instance color from buffer

                return o;
            }

            half4 frag(const varyings i) : SV_Target
            {
                if (i.color.a < 0.001f)
                    discard;
                return half4(i.color.rgb, i.color.a); // Output the color with alpha
            }
            ENDHLSL
        }
    }
}