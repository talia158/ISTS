using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using MessagePack;
using MessagePack.Resolvers;
using MessagePack.Unity;
using TMPro;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.SceneManagement;

[MessagePackObject(keyAsPropertyName: true)]
public class GraphListResponse
{
    public List<string> graphList { get; set; }
}

[System.Serializable]
public class GraphRequestBody
{
    public string graph_type;
    public int graph_seed;
    public string graph_name;
    public int n;
    public int k;
    public float beta;
    public int m;
}

[System.Serializable]
public class UserGraphRequestBody
{
    public string graph_json;
    public string graph_name;
}

public class GraphManager : MonoBehaviour
{
    public TMP_Dropdown GraphTypeDropdown;
    public TMP_InputField[] WSFields; // Expected order: N, K, Beta, Seed 
    public TMP_InputField[] BAFields; // Expected order: N, M, Seed
    public GameObject[] WSandBAObjects;
    public GameObject[] allObjects;
    public TMP_InputField nameField;
    public GameObject fileChooser;
    private string hostname = IP.hostname;
    private int port = IP.port;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        fileChooser.SetActive(false);
        SetTMPInputFieldsActive(false, WSFields);
        SetTMPInputFieldsActive(false, BAFields);
        SetGameObjectsActive(false, WSandBAObjects);

        var options = new List<string>
        {
            "Graph Model",
            "Watts Strogatz",
            "Barabasi Albert",
            "Import NetworkX JSON"
        };

        GraphTypeDropdown.AddOptions(options);

    }

    // Update is called once per frame
    void Update()
    {

    }

    public void OnGraphTypeDropdownChanged()
    {
        int value = GraphTypeDropdown.value;
        if (value == 1)
        {
            SetTMPInputFieldsActive(false, BAFields);
            SetTMPInputFieldsActive(true, WSFields);
            SetGameObjectsActive(true, WSandBAObjects);
            fileChooser.SetActive(false);
        }
        else if (value == 2)
        {
            SetTMPInputFieldsActive(false, WSFields);
            SetTMPInputFieldsActive(true, BAFields);
            SetGameObjectsActive(true, WSandBAObjects);
            fileChooser.SetActive(false);
        }
        else if (value == 3)
        {
            SetTMPInputFieldsActive(false, BAFields);
            SetTMPInputFieldsActive(false, WSFields);
            WSandBAObjects[0].SetActive(true);
            WSandBAObjects[1].SetActive(false);
            fileChooser.SetActive(true);
        }
        else
        {
            SetTMPInputFieldsActive(false, BAFields);
            SetTMPInputFieldsActive(false, WSFields);
            SetGameObjectsActive(false, WSandBAObjects);
            fileChooser.SetActive(false);
        }
    }

    public void OnHomeButtonClicked()
    {
        SceneManager.LoadScene("Home");
    }

    public void OnFileSelectionButtonClicked()
    {
        string p = PickFile();
        if (p == "")
        {
            return;
        }
        UserGraphRequestBody requestBody = new UserGraphRequestBody
        {
            graph_json = File.ReadAllText(p),
            graph_name = nameField.text
        };
        StartCoroutine(SendUserGraphPostRequest($"http://{hostname}:{port}/api/simulation/user_graph", requestBody));
    }

    public void OnGenerateButtonClicked()
    {
        try
        {
            // Hide everything
            SetGameObjectsActive(false, allObjects);
            GraphRequestBody requestBody = new GraphRequestBody
            {
                graph_type = (GraphTypeDropdown.value == 1) ? "WS" : 
                (GraphTypeDropdown.value == 2) ? "BA" : 
                null,
                graph_seed = int.Parse(WSFields[3].text),
                graph_name = nameField.text,
                n = (GraphTypeDropdown.value == 1) ? int.Parse(WSFields[0].text) : (GraphTypeDropdown.value == 2) ? int.Parse(BAFields[0].text) : 1,
                k = (GraphTypeDropdown.value == 1) ? int.Parse(WSFields[1].text) : 0,
                beta = (GraphTypeDropdown.value == 1) ? float.Parse(WSFields[2].text) : 0f,
                m = (GraphTypeDropdown.value == 2) ? int.Parse(BAFields[1].text) : 0,

            };
            // Start the coroutine to send the POST request
            StartCoroutine(SendGraphPostRequest($"http://{hostname}:{port}/api/simulation/generate_ws_ba", requestBody));
            // Navigate to simulation config page
            SceneManager.LoadScene("Config");
        }
        catch (Exception e)// Change to catch specific error! luna tips
        {
            if (!e.Message.Contains("Server Error"))
            {
                // loadingText.SetText("Client Error: " + e.Message);
            }
            else
            {
                // loadingText.SetText(e.Message);
            }
        }
    }

    private IEnumerator SendGraphPostRequest(string url, GraphRequestBody requestBody)
    {
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
        }
        else
        {
            throw new Exception("Server Error:" + request.error);
        }
    }

    private IEnumerator SendUserGraphPostRequest(string url, UserGraphRequestBody requestBody)
    {
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
        }
        else
        {
            throw new Exception("Server Error:" + request.error);
        }
    }

    // private IEnumerator FetchGraphListFromAPI()
    // {
    //     UnityWebRequest request = UnityWebRequest.Get($"http://{hostname}:{port}/api/simulation/get_heuristic_list");
    //     request.downloadHandler = new DownloadHandlerBuffer(); // Handle binary data
    //     yield return request.SendWebRequest();

    //     if (request.result == UnityWebRequest.Result.Success)
    //     {
    //         // Deserialize binary data into NodeInfo object
    //         var resolver = CompositeResolver.Create(
    //             UnityResolver.Instance,
    //             StandardResolverAllowPrivate.Instance
    //         );
    //         var options = MessagePackSerializerOptions.Standard.WithResolver(resolver);
    //         GraphListResponse heuristicListResponse = MessagePackSerializer.Deserialize<GraphListResponse>(request.downloadHandler.data, options);
    //         // Update the UI
    //         PopulateGraphDropdown(heuristicListResponse.graphList);
    //     }
    //     else
    //     {
    //         Debug.LogError($"Failed to fetch node data: {request.error}");
    //     }
    // }

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

    public string PickFile()
    {
        string filePath = string.Empty;

#if UNITY_STANDALONE_WIN
        filePath = WindowsFilePicker();
#elif UNITY_STANDALONE_OSX
        filePath = MacOSFilePicker();
#endif

        if (!string.IsNullOrEmpty(filePath))
        {
            return filePath;
        }
        return "";
    }

    private string WindowsFilePicker()
    {
        // Windows-specific file picker using native API
        OpenFileName ofn = new OpenFileName();
        ofn.structSize = Marshal.SizeOf(ofn);
        ofn.filter = "All Files\0*.*\0\0";
        ofn.file = new string(new char[256]);
        ofn.maxFile = ofn.file.Length;
        ofn.fileTitle = new string(new char[64]);
        ofn.maxFileTitle = ofn.fileTitle.Length;
        ofn.initialDir = "C:\\";
        ofn.title = "Select a File";
        ofn.defExt = "json"; // Default file extension
        ofn.flags = 0x00000008 | 0x00000004; // OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST

        if (DllMethods.GetOpenFileName(ofn))
        {
            return ofn.file;
        }

        return string.Empty;
    }

    private string MacOSFilePicker()
    {
        // macOS-specific file picker using native API (via Terminal or AppleScript)
        string result = System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
        {
            FileName = "/usr/bin/osascript",
            Arguments = "-e 'POSIX path of (choose file)'",
            RedirectStandardOutput = true,
            UseShellExecute = false,
            CreateNoWindow = true
        })?.StandardOutput.ReadToEnd().Trim();

        return result;
    }

    // Windows API declarations
    private class DllMethods
    {
        [DllImport("comdlg32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        public static extern bool GetOpenFileName([In, Out] OpenFileName ofn);
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
    public class OpenFileName
    {
        public int structSize;
        public IntPtr dlgOwner;
        public IntPtr instance;
        public string filter;
        public string customFilter;
        public int maxCustomFilter;
        public int filterIndex;
        public string file;
        public int maxFile;
        public string fileTitle;
        public int maxFileTitle;
        public string initialDir;
        public string title;
        public int flags;
        public short fileOffset;
        public short fileExtension;
        public string defExt;
        public IntPtr custData;
        public IntPtr hook;
        public string templateName;
        public IntPtr reservedPtr;
        public int reservedInt;
        public int flagsEx;
    }

}
