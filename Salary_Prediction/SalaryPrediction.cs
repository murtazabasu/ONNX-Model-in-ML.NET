using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Salary_Prediction
{
    class SalaryPrediction
    {
        [ColumnName("Score")]
        public float PredctedSalary;
    }
}
