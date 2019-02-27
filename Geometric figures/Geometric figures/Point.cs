using System;
namespace Geometric_figures
{
    public struct Point
    {
        public double X;
        public double Y;

        public Point(double x, double y)
        {
            X = x;
            Y = y;
        }

        public string GetString(int precision = 3)
        {
            return Math.Round(X, precision) + ";" + Math.Round(Y, precision);
        }
    }
}