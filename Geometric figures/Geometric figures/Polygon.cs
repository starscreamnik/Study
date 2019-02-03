using System;
using System.Collections;
using System.Collections.Generic;

namespace Geometric_figures
{
    public class Polygon : Shape
    {
        private readonly Point[] _points;
        public Polygon(Point[] points)
        {
            //if (points.Length < 3) ~Polygon();
            _points = new Point[points.Length];
            for(var i=0; i < points.Length; i++)
            {
                _points[i] = points[i];
            }
        }

        ~Polygon(){}

        public override double Area()
        {
            double res = 0;
            int curr = 0, next = 1;
            while (next < _points.Length)
            {
                res += _points[curr].X * _points[next].Y -
                       _points[next].X * _points[curr].Y;
                curr = next++;
            }

            return Math.Round(0.5 * Math.Abs(res), 3);
        }

        public override double Perimeter()
        {
            double res = 0;
            int curr = 0, next = 1;
            while (next < _points.Length)
            {
                res += GetEuclidMetric(_points[curr], _points[next]);
                curr = next++;
            }

            return Math.Round(res, 3);
        }

        public override Point Center()
        {
            double xRes = 0, yRes = 0;
            int curr = 0, next = 1;
            do
            {
                xRes += GetEuclidMetric(_points[curr], _points[next]) * _points[curr].X;
                yRes += GetEuclidMetric(_points[curr], _points[next]) * _points[curr].Y;
                curr = next++;
                next = next == _points.Length ? 0 : next;
                
            } while (curr != 0);
            return new Point(Math.Round(xRes/Perimeter(), 2), 
                             Math.Round(yRes/Perimeter(), 2));
        }
    }
}