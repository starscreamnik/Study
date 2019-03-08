using System;

namespace Ratio
{
    public class Ratio
    {
        private const double Inf = 1e+15;
        public int Numer { get; }
        public int Denom { get; }

        public Ratio(int a, int b)
        {
            if (b == 0) 
                throw new ArgumentException("Denominator is zero");
            var nod = EuclidNod(Math.Abs(a), Math.Abs(b));
            Numer = a/nod;
            Denom = b==0? b : a==0? 1 : Math.Abs(b)/nod;
        }

        private static int EuclidNod(int a, int b)
        {
            while (a != 0 && b != 0)
            {
                if (a > b)
                    a = a % b;
                else
                    b = b % a;
            }

            return a + b;
        }

        private static int Nok(int a, int b)
        {
            return (a / EuclidNod(a, b)) * b;
        }

        public static Ratio operator +(Ratio n1, Ratio n2)
        {
            if (n1.Denom == n2.Denom)
                return new Ratio(n1.Numer + n2.Numer, n1.Denom); // or n2.fraction

            var nok = Nok(n1.Denom, n2.Denom);
            var newInt = n1.Numer * nok / n1.Denom +
                         n2.Numer * nok / n2.Denom;
            return new Ratio(newInt, nok);
            
        }
        public static Ratio operator -(Ratio n1, Ratio n2)
        {
            if (n1.Denom == n2.Denom)
                return new Ratio(n1.Numer - n2.Numer, n1.Denom); // or n2.fraction

            var nok = Nok(n1.Denom, n2.Denom);
            var newInt = n1.Numer * nok / n1.Denom -
                         n2.Numer * nok / n2.Denom;
            return new Ratio(newInt, nok);

        }
        public static Ratio operator *(Ratio n1, Ratio n2)
        {
            var nod1 = EuclidNod(Math.Abs(n1.Numer), n2.Denom);
            var nod2 = EuclidNod(n1.Denom, Math.Abs(n2.Numer));
            Console.WriteLine("{0} {1}", nod1, nod2);
            return new Ratio((n1.Numer/nod1) * (n2.Numer/nod2), (n1.Denom/nod2) * (n2.Denom/nod1)); 
        }
        public static Ratio operator /(Ratio n1, Ratio n2)
        {
            if (n2.Numer == 0)
                throw new ArgumentException("Divide by zero");
            var nod1 = EuclidNod(Math.Abs(n1.Numer), Math.Abs(n2.Numer));
            var nod2 = EuclidNod(n1.Denom,n2.Denom);
            return new Ratio((n1.Numer/nod1) * (n2.Denom/nod2), (n1.Denom/nod2) * (n2.Numer/nod1));
        }

        public static Ratio operator ++(Ratio n1)
        {
            return new Ratio(n1.Numer+n1.Denom, n1.Denom);
        }

        public static Ratio operator --(Ratio n1)
        {
            return new Ratio(n1.Numer-n1.Denom, n1.Denom);
        }

        public double ToDouble(int precision = 5)
        {
            return Denom != 0 ? Math.Round((double)Numer / Denom, precision) : Inf;
        }

        public override string ToString()
        {
            return $"{Numer}/{Denom}";
        }
    }
}