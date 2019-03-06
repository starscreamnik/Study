using System;

namespace Geometric_figures
{
    public class Ellipse : Shape
    {
        public new string Name = "Ellipse";

        private readonly Point _center;
        private readonly double _a;
        private readonly double _b;
        public Ellipse(double a, double b, Point center):base("Ellipse")
        {
            _center = center;
            _a = a;
            _b = b;
        }

        public override double Area()
        {
            return Math.PI * _a * _b;
        }

        public override double Perimeter()
        {
            return 2 * Math.PI * Math.Sqrt( (_a * _a + _b * _b) * 0.5);
        }

        public override Point Center()
        {
            return _center;
        }
    }
}
