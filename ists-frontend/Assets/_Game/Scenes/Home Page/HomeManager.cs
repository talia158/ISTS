using System;
using System.Collections;
using System.Collections.Generic;
using MessagePack;
using MessagePack.Resolvers;
using MessagePack.Unity;
using TMPro;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.SceneManagement;

[MessagePackObject(keyAsPropertyName: true)]
public class SimulationList
{
    public List<string> list { get; set; }
    public List<int> idList { get; set; }
}

public class HomeManager : MonoBehaviour
{
    public Transform simulationScrollViewTransform, heuristicScrollViewTransform;
    public GameObject buttonPrefab;
    public SimulationList simulationList;
    public heuristicListResponse heuristicListResponse;
    private string hostname = IP.hostname;
    private int port = IP.port;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        StartCoroutine(FetchSimulationListFromAPI());
    }

    // Update is called once per frame
    void Update()
    {

    }

    public void OnStartClicked()
    {
        SceneManager.LoadScene("Config");
    }

    private IEnumerator FetchSimulationListFromAPI()
    {
        UnityWebRequest request = UnityWebRequest.Get($"http://{hostname}:{port}/api/simulation/list");
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
            simulationList = MessagePackSerializer.Deserialize<SimulationList>(request.downloadHandler.data, options);
            // Update the UI
            PopulateSimulationScrollView(simulationList.list);
        }
        else
        {
            Debug.LogError($"Failed to fetch node data: {request.error}");
        }
    }

    private void PopulateSimulationScrollView(List<string> simulations)
    {
        // Clear existing items in the scroll view content
        foreach (Transform child in simulationScrollViewTransform)
        {
            Destroy(child.gameObject);
        }

        // Dynamically instantiate a button for each neighbor
        int i = 0;
        foreach (string neighborLabel in simulations)
        {
            // Instantiate the button prefab
            GameObject button = Instantiate(buttonPrefab, simulationScrollViewTransform);

            // Set the button's text
            TextMeshProUGUI buttonText = button.GetComponentInChildren<TextMeshProUGUI>();
            if (buttonText != null)
            {
                buttonText.text = neighborLabel;
            }

            // Add a click listener to the button
            UnityEngine.UI.Button uiButton = button.GetComponent<UnityEngine.UI.Button>();
            if (uiButton != null)
            {
                int id = simulationList.idList[i];
                uiButton.onClick.AddListener(() => OnSimulationButtonClicked(id));
                i++;
            }
        }
    }

    private void OnSimulationButtonClicked(int id)
    {
       GraphScript.simulationID = id;
       SceneManager.LoadScene("Graph");
    }

    public void OnHeuristicButtonClicked()
    {
       SceneManager.LoadScene("Heuristic Performance");
    }
}
