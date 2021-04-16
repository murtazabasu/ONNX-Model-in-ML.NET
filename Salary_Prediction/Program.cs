using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;

namespace Salary_Prediction
{
    class Program
    {
        private static readonly string MODEL_NAME = "model.onnx";
        //private static readonly string DATA_PATH = Path.Combine("Dataset", "Salary_Data.csv");
        private static readonly string DATA_PATH = @"D:\ML\machine_learning_projects\visual_studio\ONNX Model in ML.NET\Salary_Prediction\Dataset\Salary_Data.csv";
        private static readonly string MODEL_PATH = @"D:\ML\machine_learning_projects\visual_studio\ONNX Model in ML.NET\Salary_Prediction\";

        static void Main(string[] args)
        {
            var context = new MLContext();

            var textLoader = context.Data.CreateTextLoader(new[]
            {
                new TextLoader.Column("YearsExperience", DataKind.Single, 0),
                new TextLoader.Column("Salary", DataKind.Single, 1)
            },
            hasHeader: true,
            separatorChar: ','
            );

            var data = textLoader.Load(DATA_PATH);
            data = context.Data.ShuffleRows(data);

            var trainTestData = context.Data.TrainTestSplit(data);

            var pipeline = context.Transforms.Concatenate("Features", "YearsExperience")
                .Append(context.Regression.Trainers.Sdca(labelColumnName: "Salary"));

            ITransformer model = pipeline.Fit(trainTestData.TrainSet);

            using (var stream = File.Create(MODEL_PATH + MODEL_NAME))
            {
                context.Model.ConvertToOnnx(model, data, stream);
            }

            // Load the onnx model for making predictions
            var predictor = context.Transforms.ApplyOnnxModel(MODEL_PATH + MODEL_NAME);

            var newModel = predictor.Fit(trainTestData.TestSet);
        }
    }
}
