using System;
using System.Diagnostics;
using System.IO;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Host;
using TensorFlow;

namespace CustomVisionFunctions
{
    public static class CustomVisionFunction
    {
        [FunctionName("CustomVisionFunction")]
        public static void Run([BlobTrigger("images/{name}", Connection = "ImagesStorage")]Stream myBlob, string name, TraceWriter log)
        {
            log.Info($"C# Blob trigger function Processed blob\n Name:{name} \n Size: {myBlob.Length} Bytes");

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            var graph = new TFGraph();

            var model = File.ReadAllBytes("Assets/model.pb");
            var labels = File.ReadAllLines("Assets/labels.txt");
            graph.Import(model);

            log.Info($"{name}");

            using (var session = new TFSession(graph))
            {
                var tensor = ImageUtil.CreateTensorFromImageFile(myBlob);
                var runner = session.GetRunner();
                runner.AddInput(graph["Placeholder"][0], tensor).Fetch(graph["loss"][0]);
                //    runner.AddInput(graph["input"][0], tensor).Fetch(graph["final_result"][0]);
                var output = runner.Run();
                var result = output[0];
                var threshold = 0.25; // 25%

                var probabilities = ((float[][])result.GetValue(jagged: true))[0];
                for (int i = 0; i < probabilities.Length; i++)
                {
                    // output the tags over the threshold
                    if (probabilities[i] > threshold)
                    {
                        log.Info($"{labels[i]} ({Math.Round(probabilities[i] * 100.0, 2)}%)");
                    }
                }
            }

            stopwatch.Stop();
            log.Info($"Total time: {stopwatch.Elapsed}");
        }
    }
}
