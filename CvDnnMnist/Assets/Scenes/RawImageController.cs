using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.IO;

using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.DnnModule;

public class RawImageController : MonoBehaviour
{
    private Texture2D texture;
    GameObject answerText;

    private List<List<byte>> mnist_dataset;
    private int mnist_dataset_idx;

    private Net net;

    private const int IMG_WIDTH = 28;
    private const int IMG_HEIGHT = 28;
    private const string MODEL_FILE_PATH = "mnist.pb";


    // Start is called before the first frame update
    void Start()
    {
        answerText = GameObject.Find("Text");

        mnist_dataset_idx = 0;

        mnist_dataset = read_mnist_dataset();

        string model_filepath = Utils.getFilePath(MODEL_FILE_PATH);
        net = Dnn.readNetFromTensorflow(model_filepath);
        if (net.empty())
        {
            Debug.LogError("model file is not loaded.");
        }

        show_image();
        predict();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void go_to_next_image()
    {
        mnist_dataset_idx += 1;
        if (mnist_dataset.Count <= mnist_dataset_idx) mnist_dataset_idx = 0;

        show_image();
        predict();
    }

    void show_image()
    {
        List<byte> mnist_data = mnist_dataset[mnist_dataset_idx];

        Mat disp_img = new Mat(IMG_HEIGHT, IMG_WIDTH, CvType.CV_8UC3);
        for(int y = 0; y < IMG_HEIGHT; y++)
        {
            for(int x = 0; x < IMG_WIDTH; x++)
            {
                byte p = mnist_data[y * IMG_HEIGHT + x];
                disp_img.put(y, x, new byte[3] { p, p, p });
            }
        }

        texture = new Texture2D(IMG_WIDTH, IMG_HEIGHT);
        Utils.matToTexture2D(disp_img, texture);
        GetComponent<RawImage>().texture = texture;

        /*
        //Matを使用せずtextureに画素を格納する
        texture = new Texture2D(IMG_WIDTH, IMG_HEIGHT);
        List<float> data = mnist_data[1];
        List<List<float>> d = convert_data(data, true);
        Debug.Log(d.Count + " , " + d[0].Count);

        for(int y = 0; y < IMG_HEIGHT; y++)
        {
            for(int x = 0; x < IMG_WIDTH; x++)
            {
                texture.SetPixel(x, y, new Color(d[y][x], d[y][x], d[y][x]));
            }
        }
        texture.Apply(false);

        GetComponent<RawImage>().texture = texture;
        */
    }

    void predict()
    {
        List<byte> mnist_data = mnist_dataset[mnist_dataset_idx];

        Mat input_img = new Mat(1, IMG_HEIGHT * IMG_WIDTH, CvType.CV_32FC1);
        for(int i = 0; i < mnist_data.Count; i++)
        {
            float p = (float)mnist_data[i] / 255.0f;
            input_img.put(0, i, p);
        }

        Mat blob = Dnn.blobFromImage(input_img);

        net.setInput(blob);
        Mat prob = net.forward();

        (int max_idx, float max_value) = get_max_idx(prob);
        answerText.GetComponent<Text>().text = "idx : " + max_idx + " , value : " + max_value.ToString("F2");
    }

    List<List<byte>> read_mnist_dataset()
    {
        TextAsset csv = Resources.Load("mnist_dataset") as TextAsset;
        StringReader reader = new StringReader(csv.text);
        List<List<byte>> data = new List<List<byte>>();

        while(reader.Peek() != -1)
        {
            string[] str_line = reader.ReadLine().Split(',');
            List<byte> line = new List<byte>();
            foreach(string str in str_line)
            {
                line.Add(byte.Parse(str));
            }
            data.Add(line);
        }
        return data;
    }

    List<List<float>> convert_data( List<float> data_array,
                                    bool is_texture_axis=false)
    {
        List<List<float>> d = new List<List<float>>();
        for (int i = 0; i < IMG_HEIGHT; i++) d.Add(new List<float>());
        
        for(int y = 0; y < IMG_HEIGHT; y++)
        {
            for(int x = 0; x < IMG_WIDTH; x++)
            {
                if (is_texture_axis)
                {
                    d[IMG_HEIGHT - y -1].Add(data_array[y * IMG_HEIGHT + x]);
                }
                else
                {
                    d[y].Add(data_array[y * IMG_HEIGHT + x]);
                }
            }
        }
        return d;
    }

    (int idx, float value) get_max_idx(Mat prob)
    {
        int max_idx = 0;
        float max_value = 0.0f;

        for(int i = 0; i < prob.width(); i++)
        {
            float tmp = (float)prob.get(0, i)[0];
            if(max_value < tmp)
            {
                max_value = tmp;
                max_idx = i;
            }
        }
        return (max_idx, max_value);

    }
}
