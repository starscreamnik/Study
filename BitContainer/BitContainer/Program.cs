using System;

namespace BitContainer
{
    class Program
    {
        static void Main(string[] args)
        {
            var b = new BitContainer();
            
            for (var i = 0; i < 8; i++)
                b.PushBit(i%2);
            Console.WriteLine("ADD 8 bits, Bytes: {0}", b.Length/8);
            foreach (var bit in b.MyEnumerator())
            {
                Console.Write("{0} ", bit);
            }
            Console.WriteLine();

            
            for (var i = 0; i < 7; i++)
                b.PushBit(false);
            Console.WriteLine("ADD 7 bits");
            Console.WriteLine(b.ToString());


            
            b.Insert(5, 1);
            Console.WriteLine("INSERT true TO 5");
            Console.WriteLine(b.ToString());
            
            
            b.Insert(1, false);
            Console.WriteLine("INSERT false TO 1");
            Console.WriteLine(b.ToString());

            
           
            b.Remove(0);
            Console.WriteLine("REMOVE FROM 0");
            Console.WriteLine(b.ToString());


//            b.Insert(-1, false);
//            b.Insert(b.Length, true);
//            b.Remove(20);
            
            b.Clear();
            Console.WriteLine("CLEAR");
            Console.WriteLine(b.ToString());

        }
        
    }
}