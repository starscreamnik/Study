using System;

namespace Geometric_figures
{
    public abstract class Shape
    {
        protected static double GetEuclidMetric(Point p1, Point p2)
        {
            return Math.Sqrt((p1.X - p2.X) * (p1.X - p2.X) +
                             (p1.Y - p2.Y) * (p1.Y - p2.Y));
        }
        public abstract double Area();
        public abstract double Perimeter();
        public abstract Point Center();
    }
}