using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Salary_Prediction
{
    public class SalaryData
    {
        [LoadColumn(0)]
        public float YearsExperience;

        [LoadColumn(1)]
        public float Salary;
    }

}
