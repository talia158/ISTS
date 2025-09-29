using UnityEngine;
using System.Collections.Generic;
using System.Collections;
using UnityEngine.Networking;

using MessagePack;
using MessagePack.Resolvers;
using MessagePack.Unity;
using TMPro;
using UnityEngine.UI;
using Button = UnityEngine.UI.Button;
using UnityEngine.SceneManagement;
using System;
using UnityEngine.UIElements;
using Unity.VisualScripting;
using System.Runtime.InteropServices;
using System.Linq;
using Unity.Entities.UniversalDelegates;

[StructLayout(LayoutKind.Sequential)]
public struct IntPair
{
    public int Item1;
    public int Item2;

    public IntPair(int item1, int item2)
    {
        Item1 = item1;
        Item2 = item2;
    }
}

[MessagePackObject(keyAsPropertyName: true)]
public class ApiResponse
{
    public Vector2[] circlePosList { get; set; }
    public float[] nodeScaleList { get; set; }
    public Vector2[] startPosList { get; set; }
    public Vector2[] endPosList { get; set; }
    public Color[] edgeColorList { get; set; }
    public List<(int, int)> edgeIDList { get; set; }
    public int maxUnclusteredNodePosition { get; set; }
    public int maxClusteredNodePosition { get; set; }
    public int maxClusteredEdgePosition { get; set; }
    public int maxFrame { get; set; }
    public float parentScaleFactor { get; set; }
    public int[] blacklistList { get; set; }
    public int[] nodeToClusterList { get; set; }
    public int[] clusterSizeList { get; set; }
    public int[] nodeToActiveTimestampList { get; set; }
    public uint[] bitmapList { get; set; }
    public int[] bitmapIndex { get; set; }
    public string graphID { get; set; }
}

[MessagePackObject(keyAsPropertyName: true)]
public class NodeInfo
{
    public string label { get; set; }
    public List<string> neighbourLabelList { get; set; }
    public List<int> neighbourIndexList { get; set; }
    public int clusterSize { get; set; }
    public int noActive { get; set; }
}

public class SubGraph
{
    public ApiResponse ResponseData { get; private set; }
    public NodeInfo NodeInfoData { get; private set; }
    public ComputeShader computeShader;
    public int depthInSubgraphTree;
    private ComputeBuffer _nodeToClusterBuffer, _nodeActivationBuffer, _clusterSizeBuffer, _clusterActiveCountBuffer, _clusterActivationFractionBuffer;
    private ComputeBuffer _lineStartBuffer, _lineEndBuffer, _circlePositionBuffer, _nodeScaleBuffer, _edgeColorBuffer;
    private ComputeBuffer _argsBuffer, _argsBuffer1;
    private ComputeBuffer _blacklistBuffer, _edgeIDBuffer;
    private ComputeBuffer _bitmapListBuffer, _bitmapIndexBuffer;
    private GraphicsBuffer _edgeCommandBuffer, _nodeCommandBuffer;
    private Mesh edgeMesh, nodeMesh;
    public Material edgeMaterial, nodeMaterial;
    public Dictionary<int, SubGraph> blacklistMap;
    public int[] blacklistList;
    private int cNodesCount = 0;
    private int cEdgesCount = 0;
    private int allNodesCount = 0;
    private int allEdgesCount = 0;
    private int frame = 0;
    private Camera mainCamera;
    private bool isRoot = false;
    private float nodeSF, edgeSF;

    public void Initialize(ApiResponse responseData, ComputeShader computeShader, Camera mainCamera, bool isRoot, int depthInSubgraphTree, int frame)
    {
        this.ResponseData = responseData;
        this.computeShader = UnityEngine.Object.Instantiate(computeShader);
        this.mainCamera = mainCamera;
        this.blacklistMap = new Dictionary<int, SubGraph>();
        this.blacklistList = ResponseData.blacklistList;
        this.isRoot = isRoot;
        this.depthInSubgraphTree = depthInSubgraphTree;
        this.frame = frame;

        InitializeMaterials();
        InitializeMeshes();
        InitializeBuffers();
        SetComputeShaderParameters();
        DispatchComputeShaders();
        SetMaterialBuffers();
        ConfigureIndirectDrawArguments();
        // String s = String.Join("; ", this.blacklistList);
        // Debug.Log(s);
    }

    public void addNodeToBlacklist(int nodeID, SubGraph childSubGraph)
    {
        if (nodeID < this.blacklistList.Length && this.blacklistList[nodeID] == 0)
        {
            this.blacklistMap.Add(nodeID, childSubGraph);
            this.blacklistList[nodeID] = 1;
            // update blacklist buffers 
            updateBlacklistBuffers();
        }
    }

    public void ReleaseBuffers()
    {
        _lineStartBuffer?.Release();
        _lineEndBuffer?.Release();
        _circlePositionBuffer?.Release();

        _nodeScaleBuffer?.Release();
        _edgeColorBuffer?.Release();
        _argsBuffer?.Release();
        _argsBuffer1?.Release();
        _nodeToClusterBuffer?.Release();
        _nodeActivationBuffer?.Release();
        _clusterSizeBuffer?.Release();
        _clusterActiveCountBuffer?.Release();
        _clusterActivationFractionBuffer?.Release();
        _edgeCommandBuffer?.Release();
        _nodeCommandBuffer?.Release();

        _blacklistBuffer?.Release();
        _edgeIDBuffer?.Release(); 
        
        _bitmapIndexBuffer?.Release();
        _bitmapListBuffer?.Release();

        _lineStartBuffer = null;
        _lineEndBuffer = null;
        _circlePositionBuffer = null;
        _nodeScaleBuffer = null;
        _edgeColorBuffer = null;
        _argsBuffer = null;
        _argsBuffer1 = null;
        _nodeToClusterBuffer = null;
        _nodeActivationBuffer = null;
        _clusterSizeBuffer = null;
        _clusterActiveCountBuffer = null;
        _clusterActivationFractionBuffer = null;
        _edgeCommandBuffer = null;
        _nodeCommandBuffer = null;
        _blacklistBuffer = null;
        _edgeIDBuffer = null;


        _bitmapIndexBuffer = null;
        _bitmapListBuffer = null;
    }

    private void InitializeMaterials()
    {
        edgeMaterial = new Material(Shader.Find("Talia/RectangleShader"));
        nodeMaterial = new Material(Shader.Find("Talia/EllipseShader"));
    }

    private void InitializeMeshes()
    {
        nodeMesh = CreateCircleMesh(0.01f, 6);
        edgeMesh = CreateSquareMesh();
    }

    private void InitializeBuffers()
    {
        allNodesCount = ResponseData.circlePosList.Length;
        allEdgesCount = ResponseData.startPosList.Length;

        cNodesCount = ResponseData.maxClusteredNodePosition;
        cEdgesCount = ResponseData.maxClusteredEdgePosition;

        _lineStartBuffer = new ComputeBuffer(allEdgesCount, sizeof(float) * 2);
        _lineEndBuffer = new ComputeBuffer(allEdgesCount, sizeof(float) * 2);
        _circlePositionBuffer = new ComputeBuffer(allNodesCount, sizeof(float) * 2);
        _nodeScaleBuffer = new ComputeBuffer(allNodesCount, sizeof(float));
        _edgeColorBuffer = new ComputeBuffer(allEdgesCount, sizeof(float) * 4);

        _nodeToClusterBuffer = new ComputeBuffer(ResponseData.nodeToClusterList.Length, sizeof(int));
        _nodeActivationBuffer = new ComputeBuffer(ResponseData.nodeToActiveTimestampList.Length, sizeof(int));
        _clusterSizeBuffer = new ComputeBuffer(ResponseData.clusterSizeList.Length, sizeof(int));
        _clusterActiveCountBuffer = new ComputeBuffer(ResponseData.clusterSizeList.Length, sizeof(int));
        _clusterActivationFractionBuffer = new ComputeBuffer(ResponseData.clusterSizeList.Length, sizeof(float));

        _blacklistBuffer = new ComputeBuffer(ResponseData.blacklistList.Length, sizeof(int));
        _edgeIDBuffer = new ComputeBuffer(allEdgesCount, sizeof(int) * 2);

        _bitmapIndexBuffer = new ComputeBuffer(ResponseData.bitmapIndex.Length, sizeof(int));
        _bitmapListBuffer = new ComputeBuffer(ResponseData.bitmapList.Length, sizeof(uint));

        _argsBuffer = new ComputeBuffer(1, 5 * sizeof(uint), ComputeBufferType.IndirectArguments);
        _argsBuffer1 = new ComputeBuffer(1, 5 * sizeof(uint), ComputeBufferType.IndirectArguments);

        SetBufferData();
    }

    private void ConfigureIndirectDrawArguments()
    {
        // Initialize the command buffer for edges
        _edgeCommandBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 1, GraphicsBuffer.IndirectDrawIndexedArgs.size);
        GraphicsBuffer.IndirectDrawIndexedArgs edgeCommand = new GraphicsBuffer.IndirectDrawIndexedArgs
        {
            indexCountPerInstance = edgeMesh.GetIndexCount(0),
            instanceCount = (uint)ResponseData.startPosList.Length, // Number of edges to draw
            startIndex = 0,
            startInstance = 0
        };
        _edgeCommandBuffer.SetData(new[] { edgeCommand });

        // Initialize the command buffer for nodes
        _nodeCommandBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 1, GraphicsBuffer.IndirectDrawIndexedArgs.size);
        GraphicsBuffer.IndirectDrawIndexedArgs nodeCommand = new GraphicsBuffer.IndirectDrawIndexedArgs
        {
            indexCountPerInstance = nodeMesh.GetIndexCount(0),
            instanceCount = (uint)allNodesCount, // Number of nodes to draw
            startIndex = 0,
            startInstance = 0
        };
        _nodeCommandBuffer.SetData(new[] { nodeCommand });
    }

    private List<IntPair> convertToBittableFormat(List<(int,int)> input)
    {
        List<IntPair> structList = new List<IntPair>();
        foreach (var tuple in input)
        {
            // Debug.Log(tuple.Item1.ToString() + "," + tuple.Item2.ToString());
            structList.Add(new IntPair(tuple.Item1, tuple.Item2));
        }
        return structList;
    }

    private void SetBufferData()
    {
        _lineStartBuffer.SetData(ResponseData.startPosList);
        _lineEndBuffer.SetData(ResponseData.endPosList);
        _circlePositionBuffer.SetData(ResponseData.circlePosList);
        _nodeScaleBuffer.SetData(ResponseData.nodeScaleList);
        _edgeColorBuffer.SetData(ResponseData.edgeColorList);

        _nodeToClusterBuffer.SetData(ResponseData.nodeToClusterList);
        _nodeActivationBuffer.SetData(ResponseData.nodeToActiveTimestampList);
        _clusterSizeBuffer.SetData(ResponseData.clusterSizeList);

        _blacklistBuffer.SetData(this.blacklistList);
        _edgeIDBuffer.SetData(convertToBittableFormat(ResponseData.edgeIDList));

        _bitmapIndexBuffer.SetData(ResponseData.bitmapIndex);
        _bitmapListBuffer.SetData(ResponseData.bitmapList);

        // We can change these parameters if we don't intend c/uc switching
        uint[] args = {
            edgeMesh.GetIndexCount(0),
            (uint)ResponseData.startPosList.Length,
            edgeMesh.GetIndexStart(0),
            edgeMesh.GetBaseVertex(0),
            0
        };
        uint[] args1 = {
            nodeMesh.GetIndexCount(0),
            (uint)ResponseData.circlePosList.Length,
            nodeMesh.GetIndexStart(0),
            nodeMesh.GetBaseVertex(0),
            0
        };

        _argsBuffer.SetData(args);
        _argsBuffer1.SetData(args1);
    }

    private void SetMaterialBuffers()
    {
        // Edges
        edgeMaterial.SetBuffer("position_buffer_1", _lineStartBuffer);
        edgeMaterial.SetBuffer("position_buffer_2", _lineEndBuffer);
        edgeMaterial.SetBuffer("color_buffer", _edgeColorBuffer);

        edgeMaterial.SetBuffer("blacklist_buffer", _blacklistBuffer);
        edgeMaterial.SetBuffer("edge_id_buffer", _edgeIDBuffer);

        edgeMaterial.SetBuffer("bitmap_list", _bitmapListBuffer);
        edgeMaterial.SetBuffer("bitmap_index", _bitmapIndexBuffer);

        edgeMaterial.SetInteger("_Timestep", frame);
        edgeMaterial.SetInteger("_CEdgesCount", cEdgesCount);

        // Nodes (remove nodeUCorC)
        nodeMaterial.SetBuffer("position_buffer_1", _circlePositionBuffer);
        nodeMaterial.SetBuffer("scale_buffer", _nodeScaleBuffer);
        // float[] temp = new float[ResponseData.clusterSizeList.Length];
        // _clusterActivationFractionBuffer.GetData(temp);
        // Debug.Log("Here");
        // foreach (float t in temp)
        // {
        //     Debug.Log(t);
        // }
        nodeMaterial.SetBuffer("clusterActivationFractionBuffer", _clusterActivationFractionBuffer);
        nodeMaterial.SetBuffer("nodeActivationBuffer", _nodeActivationBuffer);

        nodeMaterial.SetBuffer("blacklist_buffer", _blacklistBuffer);

        nodeMaterial.SetInteger("_Timestep", frame);
        nodeMaterial.SetInteger("_CNodesCount", cNodesCount);
    }

    private void SetComputeShaderParameters()
    {
        int kernelActiveCount = computeShader.FindKernel("NodeClusterActiveCount");
        int kernelComputeFractions = computeShader.FindKernel("ComputeClusterFractions");

        computeShader.SetInt("totalNodes", ResponseData.nodeToActiveTimestampList.Length);//Total UC Nodes
        computeShader.SetInt("totalClusters", cNodesCount);
        computeShader.SetInt("timestep", frame);

        computeShader.SetBuffer(kernelActiveCount, "nodeToClusterBuffer", _nodeToClusterBuffer);
        computeShader.SetBuffer(kernelActiveCount, "nodeActivationBuffer", _nodeActivationBuffer);
        computeShader.SetBuffer(kernelActiveCount, "clusterActiveCountBuffer", _clusterActiveCountBuffer);

        computeShader.SetBuffer(kernelComputeFractions, "clusterActiveCountBuffer", _clusterActiveCountBuffer);
        computeShader.SetBuffer(kernelComputeFractions, "clusterSizeBuffer", _clusterSizeBuffer);
        computeShader.SetBuffer(kernelComputeFractions, "clusterActivationFractionBuffer", _clusterActivationFractionBuffer);
    }

    private void DispatchComputeShaders()
    {
        int kernelActiveCount = computeShader.FindKernel("NodeClusterActiveCount");
        int kernelComputeFractions = computeShader.FindKernel("ComputeClusterFractions");
        // BUG HERE
        int groups = Mathf.CeilToInt(ResponseData.nodeToActiveTimestampList.Length / 256.0f);
        computeShader.Dispatch(kernelActiveCount, groups, 1, 1);

        groups = Mathf.CeilToInt(cNodesCount / 256.0f);
        computeShader.Dispatch(kernelComputeFractions, groups, 1, 1);
    }

    public void UpdateFrameDependentBuffers(int frame)
    {
        this.frame = frame;
        nodeMaterial.SetInteger("_Timestep", frame);
        computeShader.SetInt("timestep", frame);

        int kernelActiveCount = computeShader.FindKernel("NodeClusterActiveCount");
        int kernelComputeFractions = computeShader.FindKernel("ComputeClusterFractions");

        // Reset cluster active count
        _clusterActiveCountBuffer.SetData(new int[cNodesCount]);

        // Rebind buffers 
        computeShader.SetBuffer(kernelActiveCount, "nodeToClusterBuffer", _nodeToClusterBuffer);
        computeShader.SetBuffer(kernelActiveCount, "nodeActivationBuffer", _nodeActivationBuffer);
        computeShader.SetBuffer(kernelActiveCount, "clusterActiveCountBuffer", _clusterActiveCountBuffer);

        // Dispatch to recalc active counts (BUG HERE)
        int groups = Mathf.CeilToInt(ResponseData.nodeToActiveTimestampList.Length / 256.0f);
        computeShader.Dispatch(kernelActiveCount, groups, 1, 1);

        // Rebind and dispatch for cluster fractions
        computeShader.SetBuffer(kernelComputeFractions, "clusterActiveCountBuffer", _clusterActiveCountBuffer);
        computeShader.SetBuffer(kernelComputeFractions, "clusterSizeBuffer", _clusterSizeBuffer);
        computeShader.SetBuffer(kernelComputeFractions, "clusterActivationFractionBuffer", _clusterActivationFractionBuffer);

        groups = Mathf.CeilToInt(cNodesCount / 256.0f);
        computeShader.Dispatch(kernelComputeFractions, groups, 1, 1);

        edgeMaterial.SetInteger("_Timestep", frame);
    }    

    public void DrawMeshes()
    {
        RenderParams edgeRenderParams = new RenderParams(edgeMaterial)
        {
            worldBounds = new Bounds(Vector3.zero, Vector3.one * 10000),
            lightProbeUsage = UnityEngine.Rendering.LightProbeUsage.Off,
            shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off,
            receiveShadows = false
        };

        Graphics.RenderMeshIndirect(
            edgeRenderParams,
            edgeMesh,
            _edgeCommandBuffer,
            1,
            0
        );

        RenderParams nodeRenderParams = new RenderParams(nodeMaterial)
        {
            worldBounds = new Bounds(Vector3.zero, Vector3.one * 10000),
            lightProbeUsage = UnityEngine.Rendering.LightProbeUsage.Off,
            shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off,
            receiveShadows = false
        };

        Graphics.RenderMeshIndirect(
            nodeRenderParams,
            nodeMesh,
            _nodeCommandBuffer,
            1,
            0
        );
    }

    public int ClosestNode()
    {
        Vector3 cursorScreenPosition = Input.mousePosition;
        Vector3 cursorWorldPosition = mainCamera.ScreenToWorldPoint(new Vector3(
            cursorScreenPosition.x,
            cursorScreenPosition.y,
            Mathf.Abs(mainCamera.transform.position.z)
        ));

        Vector2[] nodePositions = ResponseData.circlePosList;
        int closestNodeIndex = -1;
        float closestDistance = float.MaxValue;

        Plane[] frustumPlanes = GeometryUtility.CalculateFrustumPlanes(mainCamera);

        int upper = nodePositions.Length;

        if (isRoot)
        {
            upper = cNodesCount;
        }

        for (int i = 0; i < upper; i++)
        {
            Vector2 nodePosition = nodePositions[i];
            Vector3 nodeWorldPosition = new Vector3(nodePosition.x, nodePosition.y, 0);

            if (!GeometryUtility.TestPlanesAABB(frustumPlanes, new Bounds(nodeWorldPosition, Vector3.one)))
            {
                continue;
            }

            float distance = Vector2.Distance(cursorWorldPosition, nodePosition);
            if (distance < closestDistance)
            {
                closestDistance = distance;
                closestNodeIndex = i;
            }
        }

        return closestNodeIndex;
    }
    private void updateBlacklistBuffers()
    {
        _blacklistBuffer?.Release();
        _blacklistBuffer = null;
        _blacklistBuffer = new ComputeBuffer(ResponseData.blacklistList.Length, sizeof(int));
        _blacklistBuffer.SetData(this.blacklistList);
        nodeMaterial.SetBuffer("blacklist_buffer", _blacklistBuffer);
        edgeMaterial.SetBuffer("blacklist_buffer", _blacklistBuffer);
    }

    private Mesh CreateCircleMesh(float circleRadius, int numSegments)
    {
        Mesh mesh = new Mesh();
        Vector3[] vertices = new Vector3[numSegments + 1];
        int[] triangles = new int[numSegments * 3];

        vertices[0] = Vector3.zero;
        Quaternion rotation = Quaternion.Euler(0f, 180f, 0f);

        for (int i = 0; i < numSegments; i++)
        {
            float angle = i * 2 * Mathf.PI / numSegments;
            Vector3 vertexPosition = new Vector3(
                Mathf.Cos(angle) * circleRadius,
                Mathf.Sin(angle) * circleRadius,
                0f
            );
            vertexPosition = rotation * vertexPosition;
            vertices[i + 1] = vertexPosition;

            if (i < numSegments - 1)
            {
                triangles[i * 3] = 0;
                triangles[i * 3 + 1] = i + 1;
                triangles[i * 3 + 2] = i + 2;
            }
            else
            {
                triangles[i * 3] = 0;
                triangles[i * 3 + 1] = i + 1;
                triangles[i * 3 + 2] = 1;
            }
        }

        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();

        return mesh;
    }

    private Mesh CreateSquareMesh()
    {
        Mesh mesh = new Mesh();
        Vector3[] vertices = {
            new Vector3(-0.5f, -0.5f, 0),
            new Vector3(0.5f, -0.5f, 0),
            new Vector3(-0.5f, 0.5f, 0),
            new Vector3(0.5f, 0.5f, 0)
        };

        int[] triangles = { 0, 2, 1, 2, 3, 1 };
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();
        return mesh;
    }
}

public class SubGraphManager
{
    private List<SubGraph> subGraphs;
    public SubGraph rootSubGraph;
    private ComputeShader computeShader;
    private Camera mainCamera;
    public int currentClosestNodeID;
    public SubGraph subGraphOfCurrentClosestNodeID;
    public int frame;
    private List<SubGraph> subGraphsWithClosestNode;
    private List<int> closestIDs; 
    private float nodeSFSlider, edgeSFSlider;
    public SubGraphManager(ComputeShader computeShader, Camera mainCamera)
    {
        this.computeShader = computeShader;
        this.mainCamera = mainCamera;
        this.frame = 0;
        this.subGraphs = new();
        this.subGraphsWithClosestNode = new();
        this.closestIDs = new();
    }
    public void initializeRootSubGraph(ApiResponse responseData)
    {
        rootSubGraph = new SubGraph();
        rootSubGraph.Initialize(responseData, this.computeShader, this.mainCamera, true, 1, 0);//Frame initially set to zero
        subGraphs.Add(rootSubGraph);
        this.subGraphOfCurrentClosestNodeID = rootSubGraph;
    }
    public void releaseBuffers()
    {
        foreach (SubGraph subGraph in subGraphs)
        {
            subGraph.ReleaseBuffers();
        }
    }
    public void initializeExpansionSubGraph(ApiResponse responseData, SubGraph parentSubGraph, int parentNodeID)
    {
        if (responseData.graphID == "NULL")
        {
            return;
        }
        SubGraph subGraph = new SubGraph();
        subGraph.Initialize(responseData, this.computeShader, this.mainCamera, false, parentSubGraph.depthInSubgraphTree + 1, this.frame);
        // Because node in parentSubGraph at parentNodeID got replaced with subGraph... 
        int i = 0;
        foreach (SubGraph s in this.subGraphsWithClosestNode)
        {
            s.addNodeToBlacklist(this.closestIDs[i], subGraph);
            i += 1;
        }
        subGraphs.Add(subGraph);
        // Set Scale factors to match other subgraphs here
        subGraph.nodeMaterial.SetFloat("_UniversalScaleFactor", Math.Max(0.3f, this.nodeSFSlider));
        subGraph.edgeMaterial.SetFloat("_UniversalScaleFactor", Math.Max(0.1f, this.edgeSFSlider));
        // todo Passthrough same scale factor for edges

        // parentSubGraph.addNodeToBlacklist(parentNodeID, subGraph);

    }
    public void drawGraphMeshes()
    {
        foreach (SubGraph subGraph in subGraphs)
        {
            // if ((subGraphs.Count > 1 && subGraph != rootSubGraph) || (subGraphs.Count == 1)) 
            // {
            //     subGraph.DrawMeshes();
            //     // String s = String.Join("; ", subGraph.blacklistList);
            //     // if (subGraphs.Count != 1)
            //     //     Debug.Log(s);
            // }
            subGraph.DrawMeshes();
        }
    }
    public void setTime()
    {
        foreach (SubGraph subGraph in subGraphs)
        {
            subGraph.edgeMaterial.SetFloat("_UnityTime", Time.time);
        }
    }
    public void changeFrame(int difference)
    {
        int newFrame = frame + difference;
        if (newFrame < 0)
        {
            frame = 0;
        }
        else if (newFrame > rootSubGraph.ResponseData.maxFrame)
        {
            frame = rootSubGraph.ResponseData.maxFrame;
        }
        else
        {
            frame = newFrame;
        }
        foreach (SubGraph subGraph in subGraphs)
        {
            subGraph.UpdateFrameDependentBuffers(frame);
        }
    }
    public void goToFrame(int frame)
    {
        int newFrame = frame;
        if (newFrame < 0)
        {
            this.frame = 0;
        }
        else if (newFrame > rootSubGraph.ResponseData.maxFrame)
        {
            this.frame = rootSubGraph.ResponseData.maxFrame;
        }
        else
        {
            this.frame = newFrame;
        }
        foreach (SubGraph subGraph in subGraphs)
        {
            subGraph.UpdateFrameDependentBuffers(this.frame);
        }
    }
    public void changeNodeScaleFactors(float newNodeScaleFactor)
    {
        // We could maintain depth of subgraph and multiply scale by 1/d for example
        foreach (SubGraph subGraph in subGraphs)
        {
            // subGraph.nodeMaterial.SetFloat("_UniversalScaleFactor", Math.Max(0.3f, 0.1f + (subGraph.ResponseData.parentScaleFactor * (3f/subGraph.depthInSubgraphTree * newNodeScaleFactor))));
            subGraph.nodeMaterial.SetFloat("_UniversalScaleFactor", Math.Max(0.3f, newNodeScaleFactor));
        }
        this.nodeSFSlider = newNodeScaleFactor;
    }
    public void changeEdgeScaleFactors(float newEdgeScaleFactor)
    {
        foreach (SubGraph subGraph in subGraphs)
        {
            // subGraph.edgeMaterial.SetFloat("_UniversalScaleFactor", 0.1f + (3f * newEdgeScaleFactor));
            subGraph.edgeMaterial.SetFloat("_UniversalScaleFactor", Math.Max(0.1f, newEdgeScaleFactor));
        }
        this.edgeSFSlider = newEdgeScaleFactor;
    }
    // public void closestNode() 
    // {
    //     // Old algorithm
    //     SubGraph currentGraph = rootSubGraph;
    //     int closestNodeID = currentGraph.ClosestNode();
    //     // Recursively check
    //     Debug.Log("Starting search...");
    //     while (currentGraph.blacklistMap.ContainsKey(closestNodeID))
    //     {
    //         Debug.Log("Recursing");
    //         SubGraph subGraphToLookIn = currentGraph.blacklistMap[closestNodeID];
    //         closestNodeID = subGraphToLookIn.ClosestNode();
    //         currentGraph = subGraphToLookIn;
    //     }
    //     this.currentClosestNodeID = closestNodeID;
    //     this.subGraphOfCurrentClosestNodeID = currentGraph;
    // }
    public void closestNode()// This is incorrect, blacklisted nodes can stil be 'discovered' so we need recursion here... 
    {
        List<Vector2> closestPoints = new();
        List<int> nodeIndexes = new();
        foreach (SubGraph subGraph in subGraphs)
        {
            int closestNodeID = subGraph.ClosestNode();
            Vector2 closestPointFromSubGraph = new Vector2(0,0);
            if (closestNodeID != -1 && subGraph.blacklistList[closestNodeID] == 0)
            {
                closestPointFromSubGraph = subGraph.ResponseData.circlePosList[closestNodeID];
            } 
            else
            {
                closestNodeID = -1;//If the second condition in the above if statement gets triggered (to make sure it gets skipped in next for loop)
            }
            closestPoints.Add(closestPointFromSubGraph);
            nodeIndexes.Add(closestNodeID);
        }
        // Now compare all these points with the cursor location
        // We need list of subGraphs and corresponding list of id's
        Vector3 cursorScreenPosition = Input.mousePosition;
        Vector3 cursorWorldPosition = mainCamera.ScreenToWorldPoint(new Vector3(
            cursorScreenPosition.x,
            cursorScreenPosition.y,
            Mathf.Abs(mainCamera.transform.position.z)
        ));
        float closestDistance = float.MaxValue;
        // int subGraphWithClosestNode = 0;
        List<SubGraph> subGraphsWithClosestNode = new();
        List<int> closestIDs = new();
        for (int i = 0; i < closestPoints.Count; i++)
        {
            if (nodeIndexes[i] == -1)
            {
                continue;
            }
            Vector2 point = closestPoints[i];
            Vector3 pointWorldPosition = new Vector3(point.x, point.y, 0);
            float distance = Vector2.Distance(cursorWorldPosition, pointWorldPosition);
            if (distance < closestDistance)
            {
                // Make new lists...
                closestDistance = distance;
                subGraphsWithClosestNode = new();
                closestIDs = new();
                subGraphsWithClosestNode.Add(subGraphs[i]);
                closestIDs.Add(nodeIndexes[i]);
                // subGraphWithClosestNode = i;
            }
            else if (distance == closestDistance)
            {
                // Update the list
                subGraphsWithClosestNode.Add(subGraphs[i]);
                closestIDs.Add(nodeIndexes[i]);
            }
        }
        // Set this to the last one, it shoudn't matter because in the backend it should map to the same node...
        this.subGraphOfCurrentClosestNodeID = subGraphsWithClosestNode[0];
        this.currentClosestNodeID = closestIDs[0];
        this.subGraphsWithClosestNode = subGraphsWithClosestNode;
        this.closestIDs = closestIDs;
    }
}

public class GraphScript : MonoBehaviour
{
    public TMP_Text loadingText;
    public TMP_Text frameText;
    public Scrollbar nodeScaleScrollbar;
    public Scrollbar edgeScaleScrollbar;
    public ApiResponse ResponseData { get; private set; }
    public NodeInfo NodeInfoData { get; private set; }
    public ComputeShader computeShader;
    public Camera mainCamera;
    public TMP_Text selectedNodeLabel;
    public TMP_Text selectedClusterSize;
    public TMP_Text infectedCount;
    public TMP_InputField frameSelector;
    public Transform neighborScrollViewTransform;
    public GameObject buttonPrefab;
    public GameObject inspectorPanel;
    private string hostname = IP.hostname;
    private int port = IP.port;
    private bool UC = false;
    private bool isInitialized = false;
    private SubGraphManager subGraphManager;

    public static int simulationID = 1;

    public void OnHomeButtonClicked()
    {
        SceneManager.LoadScene("Home");
    }

    private void Start()
    {
        StartCoroutine(FetchDataFromApi(simulationID));
    }

    private void Update()
    {
        if (!isInitialized) return;

        if (Input.GetKeyDown(KeyCode.V))
        {
            inspectorPanel.SetActive(!inspectorPanel.activeSelf);
        }
        if (Input.GetKeyDown(KeyCode.C))
        {
            DetectNodeUnderCursor();
        }
        if (Input.GetKeyDown(KeyCode.X))
        {
            ChangeFrame(1);
        }
        else if (Input.GetKeyDown(KeyCode.Z))
        {
            ChangeFrame(-1);
        }

        // if (Input.GetKeyDown(KeyCode.Space))
        // {
        //     ToggleUCMode();
        // }
        // // Temporarily disabled c/uc switching...

        if (Input.GetKeyDown(KeyCode.E))
        {
            ExpandNode();
        }


        subGraphManager.drawGraphMeshes();
        subGraphManager.setTime();
    }

    private void ExpandNode()
    {
        subGraphManager.closestNode();
        StartCoroutine(FetchExpandedNodeDataFromApi(simulationID, subGraphManager.currentClosestNodeID, subGraphManager.subGraphOfCurrentClosestNodeID));
    }

    private void ChangeFrame(int difference)
    {
        // (Last Frame) thing is not here anymore - todo...
        subGraphManager.changeFrame(difference);
        frameText.SetText("Frame: " + subGraphManager.frame);
        StartCoroutine(FetchNodeDataFromAPI(simulationID, 
        subGraphManager.currentClosestNodeID, 
        subGraphManager.subGraphOfCurrentClosestNodeID.ResponseData.graphID, 
        UC ? 1 : 0, subGraphManager.frame));
    }

    public void ResetSelection()//Whats going on here...
    {
        subGraphManager.currentClosestNodeID = 0;
        subGraphManager.subGraphOfCurrentClosestNodeID = subGraphManager.rootSubGraph;
        if (UC)
        {
            subGraphManager.currentClosestNodeID = 0;
            infectedCount.text = "";
        }
        StartCoroutine(FetchNodeDataFromAPI(simulationID, 
        subGraphManager.currentClosestNodeID, 
        subGraphManager.subGraphOfCurrentClosestNodeID.ResponseData.graphID, 
        UC ? 1 : 0, subGraphManager.frame));
    }

    public void OnNodeScrollbarChange()
    {
        if (!isInitialized) return;
        subGraphManager.changeNodeScaleFactors(nodeScaleScrollbar.value);
    }

    public void OnEdgeScrollbarChange()
    {
        if (!isInitialized) return;
        subGraphManager.changeEdgeScaleFactors(edgeScaleScrollbar.value);
    }

    public void OnGoToFrameButtonClicked()
    {
        subGraphManager.goToFrame(int.Parse(frameSelector.text));
        frameText.SetText("Frame: " + subGraphManager.frame);
        StartCoroutine(FetchNodeDataFromAPI(simulationID, 
        subGraphManager.currentClosestNodeID, 
        subGraphManager.subGraphOfCurrentClosestNodeID.ResponseData.graphID, 
        UC ? 1 : 0, subGraphManager.frame));
    }

    private void DetectNodeUnderCursor()
    {
        subGraphManager.closestNode();
        StartCoroutine(FetchNodeDataFromAPI(simulationID, 
        subGraphManager.currentClosestNodeID, 
        subGraphManager.subGraphOfCurrentClosestNodeID.ResponseData.graphID, 
        UC ? 1 : 0, subGraphManager.frame));
    }

    private IEnumerator FetchDataFromApi(int simulationID)
    {
        // subGraphManager.releaseBuffers();
        Debug.Log("Starting...");
        inspectorPanel.SetActive(false);
        loadingText.SetText("Downloading Display Data...");

        UnityWebRequest request = UnityWebRequest.Get($"http://{hostname}:{port}/api/simulation/{simulationID}/display");
        request.downloadHandler = new DownloadHandlerBuffer();
        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            loadingText.SetText("Parsing Display Data...");
            ApiResponse apiResponse = ProcessBinaryApiResponse(request.downloadHandler.data);
            loadingText.SetText("Initializing Graph...");
            subGraphManager = new SubGraphManager(computeShader, mainCamera);
            subGraphManager.initializeRootSubGraph(apiResponse);


            loadingText.gameObject.SetActive(false);
            ResetSelection();
            frameText.text = "Frame: 0";
            inspectorPanel.SetActive(true);
            isInitialized = true;
        }
        else
        {
            Debug.LogError($"Failed to fetch data: {request.error}");
        }
    }

    private IEnumerator FetchExpandedNodeDataFromApi(int simulationID, int nodeID, SubGraph subGraph)
    {
        UnityWebRequest request = UnityWebRequest.Get($"http://{hostname}:{port}/api/simulation/{simulationID}/{nodeID}/{subGraph.ResponseData.graphID}/{subGraph.depthInSubgraphTree}/expand");
        request.downloadHandler = new DownloadHandlerBuffer();
        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            ApiResponse apiResponse = ProcessBinaryApiResponse(request.downloadHandler.data);
            subGraphManager.initializeExpansionSubGraph(apiResponse, subGraph, nodeID);
        }
        else
        {
            Debug.LogError($"Failed to fetch data: {request.error}");
        }
    }

    private IEnumerator FetchNodeDataFromAPI(int simulationID, int nodeID, string subGraphID, int UCStatus, int frame)
    {
        UnityWebRequest request = UnityWebRequest.Get($"http://{hostname}:{port}/api/simulation/{simulationID}/{nodeID}/{subGraphID}/{UCStatus}/{frame}/get_info");
        request.downloadHandler = new DownloadHandlerBuffer();
        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            var resolver = CompositeResolver.Create(
                UnityResolver.Instance,
                StandardResolverAllowPrivate.Instance
            );
            var options = MessagePackSerializerOptions.Standard.WithResolver(resolver);
            NodeInfoData = MessagePackSerializer.Deserialize<NodeInfo>(request.downloadHandler.data, options);
            UpdateNodeInfoUI(NodeInfoData);
        }
        else
        {
            Debug.LogError($"Failed to fetch node data: {request.error}");
        }
    }

    private void UpdateNodeInfoUI(NodeInfo nodeInfo)
    {
        selectedNodeLabel.text = $"Label: {nodeInfo.label}";
        selectedClusterSize.text = $"Cluster Size: {nodeInfo.clusterSize}";
        PopulateScrollView(nodeInfo.neighbourLabelList);
        if (!UC)
        {
            infectedCount.text = $"Infected: {nodeInfo.noActive}";
        }
    }

    private void PopulateScrollView(List<string> neighbourLabels)
    {
        foreach (Transform child in neighborScrollViewTransform)
        {
            Destroy(child.gameObject);
        }

        int i = 0;
        foreach (string neighborLabel in neighbourLabels)
        {
            GameObject button = Instantiate(buttonPrefab, neighborScrollViewTransform);
            TextMeshProUGUI buttonText = button.GetComponentInChildren<TextMeshProUGUI>();
            if (buttonText != null)
            {
                buttonText.text = neighborLabel.ToString();
            }

            Button uiButton = button.GetComponent<Button>();
            if (uiButton != null)
            {
                int capturedLabel = NodeInfoData.neighbourIndexList[i];
                uiButton.onClick.AddListener(() => OnNeighborButtonClicked(capturedLabel));
                i++;
            }
        }
    }

    private void OnNeighborButtonClicked(int universalID)
    {
        if (universalID >= 0 && universalID < ResponseData.circlePosList.Length)
        {
            Vector2 neighborPosition = ResponseData.circlePosList[universalID];
            Vector3 newCameraPosition = new Vector3(neighborPosition.x, neighborPosition.y, mainCamera.transform.position.z);
            mainCamera.transform.position = newCameraPosition;
        }
    }

    private ApiResponse ProcessBinaryApiResponse(byte[] binaryData)
    {
        var resolver = CompositeResolver.Create(
            UnityResolver.Instance,
            StandardResolverAllowPrivate.Instance
        );
        var options = MessagePackSerializerOptions.Standard.WithResolver(resolver);
        return MessagePackSerializer.Deserialize<ApiResponse>(binaryData, options);
    }

    private void ToggleUCMode()
    {
        UC = !UC;
        ResetSelection();
    }

    private void OnDisable()
    {
        subGraphManager.releaseBuffers();
    }

    private void OnDestroy()
    {
        subGraphManager.releaseBuffers();
    }
}