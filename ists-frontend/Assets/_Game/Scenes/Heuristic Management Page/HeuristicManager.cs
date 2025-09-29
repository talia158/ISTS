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

[MessagePackObject(keyAsPropertyName: true)]
public class heuristicListResponse
{
    public List<string> heuristicList { get; set; }
}
[MessagePackObject(keyAsPropertyName: true)]
public class heuristicBodyResponse
{
    public string heuristic { get; set; }
}
[System.Serializable]
public class HeuristicUpdateRequestBody
{
    public string heuristic_name;
    public string heuristic;
}
[System.Serializable]
public class HeuristicGetRequestBody
{
    public string heuristic_name;
}
public class HeuristicManager : MonoBehaviour
{
    public TMP_Dropdown HeuristicSelectionDropdown;
    public TMP_InputField codeInputField;
    public TMP_InputField heuristicRenameField;
    private string hostname = IP.hostname;
    private int port = IP.port;
    private heuristicBodyResponse heuristicBodyResponse;
    public void OnHomeButtonClicked()
    {
        SceneManager.LoadScene("Home");
    }
    public void OnSimulationButtonClicked()
    {
        SceneManager.LoadScene("Config");
    }

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        // Update Heuristic List Dropdown
        StartCoroutine(FetchHeuristicListFromAPI());
    }

    public void OnHeuristicSelectionDropdownChanged()
    {
        int value = HeuristicSelectionDropdown.value;
        // If second or below option is selected from the dropdown...
        if (value > 0)
        {
            string stringValue = HeuristicSelectionDropdown.options[HeuristicSelectionDropdown.value].text;
            // Make POST request to the api corresponding to stringValue
            StartCoroutine(SendGetHeuristicPostRequest(stringValue));

        }
    }
    public void OnSaveButtonClicked()
    {
        StartCoroutine(SendUpdateHeuristicPostRequest(heuristicRenameField.text, codeInputField.text));
        StartCoroutine(FetchHeuristicListFromAPI());
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

    private IEnumerator SendUpdateHeuristicPostRequest(string heuristicName, string heuristicContent)
    {
        // Convert the request body to JSON
        HeuristicUpdateRequestBody requestBody = new HeuristicUpdateRequestBody
        {
            heuristic_name = heuristicName,
            heuristic = heuristicContent
        };
        string jsonBody = JsonUtility.ToJson(requestBody);

        // Create the UnityWebRequest
        UnityWebRequest request = new UnityWebRequest($"http://{hostname}:{port}/api/simulation/save_heuristic", "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonBody);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        // Send the request and wait for a response
        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            throw new Exception("Server Error:" + request.error);
        }
    }

    private IEnumerator SendGetHeuristicPostRequest(string heuristicName)
    {
        HeuristicGetRequestBody requestBody = new HeuristicGetRequestBody
        {
            heuristic_name = heuristicName,
        };
        // Convert the request body to JSON
        string jsonBody = JsonUtility.ToJson(requestBody);

        // Create the UnityWebRequest
        UnityWebRequest request = new UnityWebRequest($"http://{hostname}:{port}/api/simulation/get_heuristic", "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonBody);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        // Send the request and wait for a response
        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            var resolver = CompositeResolver.Create(
                UnityResolver.Instance,
                StandardResolverAllowPrivate.Instance
            );
            var options = MessagePackSerializerOptions.Standard.WithResolver(resolver);
            heuristicBodyResponse = MessagePackSerializer.Deserialize<heuristicBodyResponse>(request.downloadHandler.data, options);
            // Update the UI
            codeInputField.SetTextWithoutNotify(heuristicBodyResponse.heuristic);
            heuristicRenameField.SetTextWithoutNotify(heuristicName);
        }
        else
        {
            throw new Exception("Server Error:" + request.error);
        }
    }

    private void PopulateHeuristicDropdown(List<string> heuristicList)
    {
        HeuristicSelectionDropdown.ClearOptions();
        HeuristicSelectionDropdown.AddOptions(new List<string> { 
            "Add New Heuristic"
        });
        HeuristicSelectionDropdown.AddOptions(heuristicList);
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
