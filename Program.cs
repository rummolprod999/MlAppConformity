using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;

using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Conversions;
using Microsoft.ML.Transforms.Normalizers;

namespace MlAppConformity
{
    class Program
    {
        private static string _trainDataPath => Path.Combine(Environment.CurrentDirectory, "Data", "placing_way.tsv");
        private static string _testDataPath => Path.Combine(Environment.CurrentDirectory, "Data", "placing_way_test.tsv");
        private static string _modelPath => Path.Combine(Environment.CurrentDirectory, "Models", "model.zip");

        private static MLContext _mlContext;
        private static PredictionEngine<СonformChecker, CheckerPrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;
        static void Main(string[] args)
        {
            _mlContext = new MLContext(seed: 0);
            Console.WriteLine($"=============== Loading Dataset  ===============");
            _trainingDataView = _mlContext.Data.CreateTextReader<СonformChecker>(hasHeader: true).Read(_trainDataPath);
            Console.WriteLine($"=============== Finished Loading Dataset  ===============");
            var pipeline = ProcessData();
            var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);
            Evaluate();
            PredictConformity();
        }

        public static EstimatorChain<ITransformer> ProcessData()
        {
            Console.WriteLine($"=============== Processing Data ===============");
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey("Con", "Label")
                            .Append(_mlContext.Transforms.Text.FeaturizeText("Name", "NameFeaturized"))
                            .Append(_mlContext.Transforms.Concatenate("Features", "NameFeaturized"))
                            .AppendCacheCheckpoint(_mlContext);

            Console.WriteLine($"=============== Finished Processing Data ===============");
            return pipeline;
        }

        public static EstimatorChain<KeyToValueMappingTransformer> BuildAndTrainModel(IDataView trainingDataView, EstimatorChain<ITransformer> pipeline)
        {
            var trainer = new SdcaMultiClassTrainer(_mlContext, DefaultColumnNames.Label, DefaultColumnNames.Features);
            var trainingPipeline = pipeline.Append(trainer)
                    .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            Console.WriteLine($"=============== Training the model  ===============");
            _trainedModel = trainingPipeline.Fit(trainingDataView);
            Console.WriteLine($"=============== Finished Training the model Ending time: {DateTime.Now.ToString()} ===============");
            Console.WriteLine($"=============== Single Prediction just-trained-model ===============");
            _predEngine = _trainedModel.CreatePredictionEngine<СonformChecker, CheckerPrediction>(_mlContext);
            СonformChecker conf = new СonformChecker()
            {
                Name = "Электронный аукцион"
            };
            var prediction = _predEngine.Predict(conf);
            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Con} ===============");
            SaveModelAsFile(_mlContext, _trainedModel);
            return trainingPipeline;

        }

        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fs);

            Console.WriteLine("The model is saved to {0}", _modelPath);
        }

        public static void Evaluate()
        {
            Console.WriteLine($"=============== Evaluating to get model's accuracy metrics - Starting time: {DateTime.Now.ToString()} ===============");
            var testDataView = _mlContext.Data.CreateTextReader<СonformChecker>(hasHeader: true).Read(_testDataPath);
            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));
            Console.WriteLine($"=============== Evaluating to get model's accuracy metrics - Ending time: {DateTime.Now.ToString()} ===============");
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.AccuracyMicro:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.AccuracyMacro:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");
        }

        public static void PredictConformity()
        {
            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = _mlContext.Model.Load(stream);
            }

            СonformChecker singleConf = new СonformChecker() { Name = "котировка" };
            _predEngine = loadedModel.CreatePredictionEngine<СonformChecker, CheckerPrediction>(_mlContext);
            var prediction = _predEngine.Predict(singleConf);
            Console.WriteLine($"=============== Single Prediction - Result: {prediction.Con} ===============");

        }

    }
}
