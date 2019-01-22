using Microsoft.ML.Data;

namespace MlAppConformity
{
    public class СonformChecker
    {
        [LoadColumn(0)]
        public int Con { get; set; }
        [LoadColumn(1)]
        public string Name { get; set; }
    }

    public class CheckerPrediction
    {
        [ColumnName("PredictedLabel")]
        public int Con;
    }
}