using System;

namespace BitContainer
{
    class Program
    {
        static void Main(string[] args)
        {
            var b = new BitContainer();
            
            for (var i = 0; i < 8; i++)
                b.PushBit(1);
            Console.WriteLine("ADD 8 bits, Bytes: {0}", b);
            for (var i = 0; i<b.Length; i++)
                Console.Write("{0} ", b.ToString(b[i]));
            Console.WriteLine();
            
            
            for (var i = 0; i < 7; i++)
                b.PushBit(false);
            Console.WriteLine("ADD 7 bits");
            for (var i = 0; i<b.Length; i++)
                Console.Write("{0} ", b.ToString(b[i]));
            Console.WriteLine();

            
            b.Insert(5, 1);
            Console.WriteLine("INSERT true TO 5");
            foreach (var bit in b.MyEnumerator()) 
                Console.Write("{0} ", b.ToString(bit));
            Console.WriteLine();
            
            
            b.Insert(1, false);
            Console.WriteLine("INSERT false TO 1");
            foreach (var bit in b.MyEnumerator())
                Console.Write("{0} ", b.ToString(bit));
            Console.WriteLine();
            
           
            b.Remove(0);
            Console.WriteLine("REMOVE FROM 0");
            foreach (var bit in b.MyEnumerator())
                Console.Write("{0} ", b.ToString(bit));
            Console.WriteLine();

//            b.Insert(-1, false);
//            b.Insert(b.Length, true);
//            b.Remove(20);
            
            b.Clear();
            Console.WriteLine("CLEAR");
            foreach (var bit in b.MyEnumerator())
                Console.Write("{0} ", b.ToString(bit));
            Console.WriteLine();
        }
        
    }
}