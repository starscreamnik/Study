using System;

namespace Geometric_figures
{
    public class Circle : Shape
    {
        public new string  Name;

        private readonly Point _center;
        private readonly double _radius;
        public Circle(Point center, double radius):base("Circle")
        {
            _center = center;
            _radius = radius;
        }

        public override double Area()
        {
            return Math.PI * _radius * _radius;
        }

        public override double Perimeter()
        {
            return 2 * Math.PI * _radius;
        }

        public override Point Center()
        {
            return _center;
        }
    }
}