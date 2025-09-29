Shader "Talia/RectangleShader"
{
    Properties
    {
        _WhiteColor("White Color", Color) = (1, 1, 1, 1) // Fixed white color
        _Timestep ("Timestep", Integer) = 0
        _Mode ("Mode", Float) = 1 // 0 is unclustered, 1 is clustered
        _UniversalScaleFactor ("UniversalScaleFactor", Float) = 1
        _CEdgesCount ("CEdgesCount", Integer) = 0
        _UnityTime ("UnityTime", Float) = 0 
    }
    SubShader
    {
        Pass
        {
            Tags
            {
                "RenderType" = "Opaque"
                "RenderPipeline" = "UniversalRenderPipeline"
            }

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            float4 _WhiteColor; // Define the white color
            uint _Timestep;
            float _Mode;
            float _UniversalScaleFactor;
            uint _CEdgesCount;
            float _UnityTime;

            // Structured buffers for positions (start and end)
            StructuredBuffer<float2> position_buffer_1; // Start point (A)
            StructuredBuffer<float2> position_buffer_2; // End point (B)
            StructuredBuffer<float4> color_buffer;     // Color buffer (RGBA)
            StructuredBuffer<uint> blacklist_buffer; // buffer[nodeid] -> 0/1(blacklisted)
            StructuredBuffer<uint2> edge_id_buffer; // list of [node id of A, node id of B]
            StructuredBuffer<uint> bitmap_list; // Flattened bitmap list
            StructuredBuffer<uint> bitmap_index; // Flattened bitmap index

            struct attributes
            {
                float4 vertex : POSITION; // Local vertex position
                float2 uv : TEXCOORD0;   // Texture coordinates
            };

            struct varyings
            {
                float4 vertex : SV_POSITION; // Vertex position
                float4 color : COLOR;   // Vertex color
            };

            varyings vert(attributes v, const uint instance_id : SV_InstanceID)
            {
                varyings o;
                uint2 correspondingNodeIDs = edge_id_buffer[instance_id];
                uint uNodeID = correspondingNodeIDs.x;
                uint vNodeID = correspondingNodeIDs.y;
                // If 1 or more nodes are blacklisted, then don't render the edge
                if ((blacklist_buffer[uNodeID] + blacklist_buffer[vNodeID] > 0) || (_Mode > 0.1f && instance_id >= _CEdgesCount) || (_Mode <= 0.1f && instance_id < _CEdgesCount))
                {
                    o.vertex = float4(0.0f,0.0f,0.0f,0.0f);
                    o.color = float4(0.0f,0.0f,0.0f,0.0f);
                    return o;
                }
                // Retrieve start and end positions from the buffer
                float2 start = position_buffer_1[instance_id];
                float2 end = position_buffer_2[instance_id];
                float4 instanceColor = color_buffer[instance_id]; // Retrieve color from buffer

                // Bitmap stuff here
                // 1) Read how many 32-bit words this edge has
                uint pos = bitmap_index[instance_id];
                uint count = bitmap_list[pos];

                // 2) If count == 0, there are no words => no highlight
                if (count == 0) {
                    // Just skip
                }
                else {
                    // 3) If _Timestep < (count * 32), decode which 32-bit word
                    if (_Timestep < (count * 32)) {
                        // offset = which 32-bit word
                        uint offset = _Timestep / 32;  // integer division
                        // The actual word is at pos + 1 + offset
                        uint bitmap = bitmap_list[pos + 1 + offset];

                        // 4) which bit
                        uint bitIndex = _Timestep % 32;

                        // 5) highlight if set
                        if ((bitmap & (1 << bitIndex)) != 0) {
                            float t = 0.5 + 0.5 * sin(_UnityTime * 4);
                            float3 blue = float3(0, 1, 1);
                            instanceColor.rgb = lerp(instanceColor.rgb, blue, t);
                        }
                    }
                }

                // Compute the midpoint (origin)
                float2 origin = (start + end) * 0.5;

                // Compute width (distance between start and end points)
                float width = distance(start, end);

                // Compute rotation angle in the 2D plane (X-Y) 
                float angle = atan2(end.y - start.y, end.x - start.x);

                // Apply scale to the vertex
                float2 scaledVertex = float2(v.vertex.x * width, v.vertex.y * 0.01 * _UniversalScaleFactor);

                // Apply rotation around the Z-axis (2D rotation)
                float2 rotatedVertex;
                rotatedVertex.x = scaledVertex.x * cos(angle) - scaledVertex.y * sin(angle);
                rotatedVertex.y = scaledVertex.x * sin(angle) + scaledVertex.y * cos(angle);

                // Translate the vertex to the world position
                float2 world_pos = origin + rotatedVertex;

                o.vertex = mul(UNITY_MATRIX_VP, float4(world_pos, 0, 1.0f)); // Transform to clip space
                o.color = instanceColor; // Set the fixed white color

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