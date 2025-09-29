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
public class PerformanceManager : MonoBehaviour
{
    private Dictionary<string, List<string>> heuristicPerformanceDictionary;
    public TMP_Text performanceText;
    public TMP_Dropdown HeuristicSelectionDropdown;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        StartCoroutine(FetchHeuristicPerformanceData());
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    private IEnumerator FetchHeuristicPerformanceData()
    {
        UnityWebRequest request = UnityWebRequest.Get($"http://{IP.hostname}:{IP.port}/api/simulation/get_heuristic_performance_data");
        request.downloadHandler = new DownloadHandlerBuffer(); // Handle binary data
        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            // Deserialize binary data 
            var resolver = CompositeResolver.Create(
                UnityResolver.Instance,
                StandardResolverAllowPrivate.Instance
            );
            var options = MessagePackSerializerOptions.Standard.WithResolver(resolver);
            heuristicPerformanceDictionary = MessagePackSerializer.Deserialize<Dictionary<string, List<string>>>(request.downloadHandler.data, options);
            Debug.Log(heuristicPerformanceDictionary.Count);
            populateHeuristicDropdown();
            onHeuristicDropdownChanged();
        }
        else
        {
            Debug.LogError($"Failed to fetch performance data: {request.error}");
        }
    }

    private void populateHeuristicDropdown()
    {
        HeuristicSelectionDropdown.ClearOptions();
        List<string> options = new();
        // options.Add("All Heuristics"); implement this later...
        foreach (string heuristic in heuristicPerformanceDictionary.Keys)
        {
            options.Add(heuristic);
        }
        HeuristicSelectionDropdown.AddOptions(options);
    }

    public void OnHomeButtonClicked()
    {
        SceneManager.LoadScene("Home");
    }

    public void onHeuristicDropdownChanged()
    {
        // Get name of selected heuristic
        string selectedHeuristic = HeuristicSelectionDropdown.options[HeuristicSelectionDropdown.value].text;
        // Generate the string
        List<string> components = heuristicPerformanceDictionary[selectedHeuristic];
        string content = "";
        foreach (string component in components)
        {
            content = content + component + "\n";
        }
        performanceText.SetText(content);

    }
}
