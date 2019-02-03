using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using NDesk.Options;

namespace File_comparator
{
    internal class Program
    {
        [Flags]
        private enum WOptions{
            Symbol = 0x01,
            Apart = 0x02,
            Short = 0x04
        }
        private static List<byte> Get_mismatch_list(ref byte[] f1, ref byte[] f2, ref int offset)
        {
            var line = new List<byte>();
            while (f1[offset] != f2[offset])
            {
                line.Add(f1[offset]);
                line.Add(f2[offset]);
                offset++;
                if (offset >= Math.Min(f1.Length, f2.Length)) break;
            }
            return line;
        }

        private static List<byte> Get_after_eof(ref byte[] largerFile, int offset)
        {
            var res = new List<byte>();
            for (var i = offset; i<largerFile.Length; i++)
            {
                res.Add(largerFile[i]);
            }
            return res;
        }
        private static List<KeyValuePair<int, List<byte>>> Analyze(ref byte[] data1, ref byte[] data2)
        {
            var result = new List<KeyValuePair<int, List<byte>>>(); 
            
            for (var i = 0; i < Math.Min(data1.Length, data2.Length); i++)
                if (data1[i] != data2[i])
                    result.Add(new KeyValuePair<int, List<byte>>
                        (i, Get_mismatch_list(ref data1, ref data2, ref i)));
            
            if (data1.Length > data2.Length)            
                result.Add(new KeyValuePair<int, List<byte>>(-1, Get_after_eof(ref data1, data2.Length)));       
            
            else if (data1.Length < data2.Length)           
                result.Add(new KeyValuePair<int, List<byte>>(-1, Get_after_eof(ref data2, data1.Length)));
            
            return result;
        }

        private static string MakeFormat(bool setSymbol, byte b)
        {
            var format = setSymbol ? "{0}" : " 0x{0:x2} ";
            if (setSymbol)
            {
                char symbol = !Char.IsLetterOrDigit((char) b) ? '.' : (char) b;
                return String.Format(format, symbol);
            }
            else
            {
                return String.Format(format, b);
            }
        }

        private static void PrintMismatches(List<KeyValuePair<int, List<byte>>> res, WOptions opts, int sl)
        {
            // sl - symbol length
            bool setSymbol = (opts & WOptions.Symbol) == WOptions.Symbol,
                 setApart = (opts & WOptions.Apart) == WOptions.Apart;
            
            foreach (var line in res)
            {
                var eofLen = sl <= -1 || sl > line.Value.Count ? line.Value.Count : sl;
                 
                if (line.Key == -1)
                {
                    Console.Write("EOF\t\t:");
                    
                    for (int i = 0; i < eofLen; i++)
                        Console.Write(MakeFormat(setSymbol, line.Value[i]));
                    break;
                }
                
                // write offset only in Hex-type
                Console.Write("0x{0:x8}\t:", line.Key);

                // set output length of not matching pairs
                var length = sl * 2 <= -1 || sl * 2 > line.Value.Count ? line.Value.Count : sl * 2;
                var inc = setApart ? 2 : 1;
                
                for (var i = 0; i < length; i += inc)
                {
                    if(i%2 != 0) Console.Write('(');
                    Console.Write(MakeFormat(setSymbol, line.Value[i]));
                    if(i%2 != 0) Console.Write(')');
                    
                    if (i == length-2 && setApart)
                    {
                        Console.Write("|");
                        for (var j = 1; j < length; j += 2)
                            Console.Write(MakeFormat(setSymbol, line.Value[j]));
                    }
                }
                Console.WriteLine();
            }
        }
        
        public static void Main(string[] args)
        {
            var files = new string[] { null, null };
            byte[] fileData1, fileData2;
            WOptions options = 0x00;    
            int outputLength = -1;

            var parser = new OptionSet
            {
                { "l|len=", (int n) => outputLength = n>=0? n:outputLength },
                { "q|brief", v => options |= v!=null? WOptions.Short:0x00 },
                { "s|text", v => options |= v!=null? WOptions.Symbol: 0x00 },
                { "a|side-by-side", v => options |= v!=null? WOptions.Apart: 0x00 }
            };
            parser.Parse(args);
            
            try
            {
                if (args.Length >= 2)
                {
                    if (!File.Exists(args[args.Length - 2]) || !File.Exists(args[args.Length - 1]))
                        throw new ArgumentException("The input files are corrupted or path is wrong.");
                    files[0] = args[args.Length - 2];
                    files[1] = args[args.Length - 1];
                }
                else 
                        throw new ArgumentException("2 input files expected.");
            }
            catch(ArgumentException e)
            {
                Console.Write("bundling: ");
                Console.WriteLine(e.Message);
                return;
            }
            
            //Reading from files
            using (var fs = new StreamReader(files[0]))
            { fileData1 = Encoding.ASCII.GetBytes(fs.ReadToEnd()); }

            using (var fs = new StreamReader(files[1]))
            { fileData2 = Encoding.ASCII.GetBytes(fs.ReadToEnd()); }

            Console.WriteLine("File comparison");          
            
            // get List<KeyValuePair<int, List<byte>>>, where Key - offset, Value - List of unmatched bytes
            var result = Analyze(ref fileData1, ref fileData2);
            
            if (result.Count == 0)
                Console.WriteLine("Files are identical.");
            else
            {
                Console.Write("Files are different");
                
                if ((options & WOptions.Short) == WOptions.Short) 
                    Console.WriteLine("."); 
                else
                {
                    Console.WriteLine(":");
                    PrintMismatches(result, options, outputLength);
                }
            }
        }
    }
}