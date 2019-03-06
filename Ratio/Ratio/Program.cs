using System;
using System.Collections.Generic;

namespace Ratio
{
    internal static class Program
    {
        private const double Inf = 1e+15;

        private static void PrintOperations(ref Dictionary<string, Ratio> results)
        {
            Console.WriteLine("Operation \tDecimal \tFractional");
            foreach (var (op, res) in results)
                Console.WriteLine("{0}\t\t{1}\t\t{2}",
                    op,
                    res.ToDouble(2),
                    res.ToString());
        }

        public static void Main(string[] args)
        {
            var a = new Ratio(3, 4);
            var b = new Ratio(1, 6);
            Console.WriteLine("A = {0}/{1}, B = {2}/{3}", a.Numer, a.Denom, b.Numer, b.Denom);

            var results = new Dictionary<string, Ratio>
            {
                {"sum", a + b}, {"sub", a - b}, {"mul", a * b}, {"div", a / b}, {"inc", ++a}, {"dec", b}
            };
            PrintOperations(ref results);
        }
    }
}