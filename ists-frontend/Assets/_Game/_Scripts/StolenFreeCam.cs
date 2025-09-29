using UnityEngine;

public class StolenFreeCam : MonoBehaviour
{
    public float speed = 2f;
    public float freeLookSensitivity = 0.1f;
    public float zoomSpeed = 1f;
    public float minZValue = -0.2f;

    private Vector3 _cameraPosition;

    private void Update()
    {
        // Cache transform position
        _cameraPosition = transform.position;

        // Handle Movement
        HandleMovement();

        // Handle Zoom
        HandleZoom();

        // Clamp Z Position
        _cameraPosition.z = Mathf.Min(_cameraPosition.z, minZValue);
        transform.position = _cameraPosition;
    }

    private void HandleMovement()
    {
        Vector3 direction = Vector3.zero;



        // Up and Down Movement
        if (Input.GetKey(KeyCode.W))
            direction += Vector3.up;

        if (Input.GetKey(KeyCode.S))
            direction -= Vector3.up;

        if (Input.GetKey(KeyCode.A))
            direction += Vector3.left;

        if (Input.GetKey(KeyCode.D))
            direction -= Vector3.left;

        _cameraPosition += direction * speed * Time.deltaTime;
    }

    private void HandleZoom()
    {
        var scrollInput = Input.GetAxis("Mouse ScrollWheel");

        if (Mathf.Abs(scrollInput) > 0.01f)
        {
            _cameraPosition += transform.forward * scrollInput * zoomSpeed;
        }
    }
}