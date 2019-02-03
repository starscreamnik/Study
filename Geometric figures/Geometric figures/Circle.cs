using System;

namespace Geometric_figures
{
    public class Circle : Shape
    {
        private readonly Point _center;
        private readonly double _radius;
        public Circle(double x, double y, int radius)
        {
            _center = new Point(x, y);
            _radius = radius;
        }

        public override double Area()
        {
            return Math.Round(Math.PI * _radius * _radius, 3);
        }

        public override double Perimeter()
        {
            return Math.Round(2 * Math.PI * _radius, 3);
        }

        public override Point Center()
        {
            return _center;
        }
    }
}