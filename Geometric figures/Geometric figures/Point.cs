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

        public string GetString()
        {
            return "x:"+X + ", y:" + Y;
        }
    }
}