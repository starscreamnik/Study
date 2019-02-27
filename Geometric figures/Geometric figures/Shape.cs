using System;
using System.Xml;

namespace Geometric_figures
{
    public abstract class Shape
    {
        public string Name;

        protected Shape(string name)
        {
            Name = name ?? "Undefined";
        }

        protected double GetEuclidMetric(Point p1, Point p2)
        {
            return Math.Sqrt((p1.X - p2.X) * (p1.X - p2.X) +
                             (p1.Y - p2.Y) * (p1.Y - p2.Y));
        }
        public abstract double Area();
        public abstract double Perimeter();
        public abstract Point Center();
    }
}