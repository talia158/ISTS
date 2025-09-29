using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using MessagePack;
using MessagePack.Resolvers;
using MessagePack.Unity;
using TMPro;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.SceneManagement;
// We need to be able to switch between tabs and still preserve the form input - and a button to clear it too...
[System.Serializable]
public class SimulationRequestBody
{
    public string graph_name;
    public int layout_seed;
    public int simulation_seed;
    public int heuristic_k;
    public string model_name;
    public string heuristic_name;
    public int simulationID;
}

public class ConfigManager : MonoBehaviour
{
    public TMP_Dropdown GraphTypeDropdown, ModelDropdown, HeuristicSelectionDropdown;
    public TMP_InputField[] universalFields; // Expected Order: Heuristic K, Cascade Seed, Layout Seed, Simulation ID 
    public GameObject saveButton;
    public TMP_Text loadingText;
    public GameObject[] allObjects;
    private string hostname = IP.hostname;
    private int port = IP.port;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        // Initially hide all fields
        loadingText.gameObject.SetActive(false);
        GraphTypeDropdown.ClearOptions();
        ModelDropdown.ClearOptions();
        HeuristicSelectionDropdown.ClearOptions();

        var options = new List<string>
        {
            "Linear Threshold",
            "Independent Cascade"
        };

        ModelDropdown.AddOptions(options);

        StartCoroutine(FetchHeuristicListFromAPI());
        StartCoroutine(FetchGraphListFromAPI());
    }

    // Update is called once per frame
    void Update()
    {

    }

    private IEnumerator FetchHeuristicListFromAPI()
    {
        UnityWebRequest request = UnityWebRequest.Get($"http://{hostname}:{port}/api/simulation/get_heuristic_list");
        request.downloadHandler = new DownloadHandlerBuffer(); // Handle binary data
        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            // Deserialize binary data into NodeInfo object
            var resolver = CompositeResolver.Create(
                UnityResolver.Instance,
                StandardResolverAllowPrivate.Instance
            );
            var options = MessagePackSerializerOptions.Standard.WithResolver(resolver);
            heuristicListResponse heuristicListResponse = MessagePackSerializer.Deserialize<heuristicListResponse>(request.downloadHandler.data, options);
            // Update the UI
            PopulateHeuristicDropdown(heuristicListResponse.heuristicList);
        }
        else
        {
            Debug.LogError($"Failed to fetch node data: {request.error}");
        }
    }

    private void PopulateHeuristicDropdown(List<string> heuristicList)
    {
        HeuristicSelectionDropdown.ClearOptions();
        HeuristicSelectionDropdown.AddOptions(heuristicList);
    }

    private IEnumerator FetchGraphListFromAPI()
    {
        UnityWebRequest request = UnityWebRequest.Get($"http://{hostname}:{port}/api/simulation/get_graph_list");
        request.downloadHandler = new DownloadHandlerBuffer(); // Handle binary data
        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            // Deserialize binary data into NodeInfo object
            var resolver = CompositeResolver.Create(
                UnityResolver.Instance,
                StandardResolverAllowPrivate.Instance
            );
            var options = MessagePackSerializerOptions.Standard.WithResolver(resolver);
            GraphListResponse graphListResponse = MessagePackSerializer.Deserialize<GraphListResponse>(request.downloadHandler.data, options);
            // Update the UI
            PopulateGraphDropdown(graphListResponse.graphList);
        }
        else
        {
            Debug.LogError($"Failed to fetch node data: {request.error}");
        }
    }

    private void PopulateGraphDropdown(List<string> graphList)
    {
        GraphTypeDropdown.ClearOptions();
        GraphTypeDropdown.AddOptions(graphList);
    }

    private void SetTMPInputFieldsActive(bool state, TMP_InputField[] list)
    {
        foreach (TMP_InputField t in list)
        {
            t.gameObject.SetActive(state);
        }
    }

    private void SetGameObjectsActive(bool state, GameObject[] list)
    {
        foreach (GameObject g in list)
        {
            g.SetActive(state);
        }
    }

    public void OnHomeButtonClicked()
    {
        SceneManager.LoadScene("Home");
    }

    public void OnGraphButtonClicked()
    {
        SceneManager.LoadScene("Graph Page");
    }

    public void OnHeuristicButtonClicked()
    {
        SceneManager.LoadScene("Heuristic Management");
    }

    public void OnSaveButtonClicked()
    {
        // Gather all the fields, form a request and then send it to the backend
        try
        {
            // Hide everything
            SetGameObjectsActive(false, allObjects);
            // Show loading text
            loadingText.gameObject.SetActive(true);
            SimulationRequestBody requestBody = new SimulationRequestBody
            {
                graph_name = GraphTypeDropdown.options[GraphTypeDropdown.value].text,
                layout_seed = int.Parse(universalFields[2].text),
                simulation_seed = int.Parse(universalFields[1].text),
                heuristic_k = int.Parse(universalFields[0].text),
                model_name = ModelDropdown.options[ModelDropdown.value].text,
                heuristic_name = HeuristicSelectionDropdown.options[HeuristicSelectionDropdown.value].text,
                simulationID = int.Parse(universalFields[3].text),//How to determine this??
            };
            // Start the coroutine to send the POST request
            StartCoroutine(SendSimulationPostRequest($"http://{hostname}:{port}/api/simulation/run", requestBody));
        }
        catch (Exception e)// Change to catch specific error! luna tips
        {
            if (!e.Message.Contains("Server Error"))
            {
                loadingText.SetText("Client Error: " + e.Message);
            }
            else
            {
                loadingText.SetText(e.Message);
            }
        }

    }

    private IEnumerator SendSimulationPostRequest(string url, SimulationRequestBody requestBody)
    {
        // Show loading text
        loadingText.gameObject.SetActive(true);
        // Convert the request body to JSON
        string jsonBody = JsonUtility.ToJson(requestBody);

        // Create the UnityWebRequest
        UnityWebRequest request = new UnityWebRequest(url, "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonBody);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        // Send the request and wait for a response
        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            Debug.Log("Response: " + request.downloadHandler.text);
            GraphScript.simulationID = int.Parse(universalFields[3].text);
            SceneManager.LoadScene("Graph");
        }
        else
        {
            throw new Exception("Server Error:" + request.error);
        }
    }

}
