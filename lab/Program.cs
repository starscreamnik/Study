using System;
using System.Collections.Generic;
using System.IO;
using NDesk.Options;

namespace lab
{
    internal class Program
    {
        private static string Write_mismatch(ref string txt1, ref string txt2, ref int offset)
        {
            string line = null;
            while (txt1[offset] != txt2[offset])
            {
                line += txt1[offset];
                line += txt2[offset];
                offset++;
                if (offset >= Math.Min(txt1.Length, txt2.Length)) break;
            }
            return line;
        }

        private static string Write_after_eof(ref string largerFile, int offset)
        {
            return largerFile.Substring(offset);
        }
        private static List<KeyValuePair<int, string>> Analyze(ref string file1, ref string file2)
        {
            var result = new List<KeyValuePair<int, string>>(); 
            
            for (var i = 0; i < Math.Min(file1.Length, file2.Length); i++)
                if (file1[i] != file2[i])
                    result.Add(new KeyValuePair<int, string>
                        (i, Write_mismatch(ref file1, ref file2, ref i)));
            
            if (file1.Length > file2.Length)            
                result.Add(new KeyValuePair<int, string>(-1, Write_after_eof(ref file1, file2.Length)));       
            
            else if (file1.Length < file2.Length)           
                result.Add(new KeyValuePair<int, string>(-1, Write_after_eof(ref file2, file1.Length)));
            
            return result;
        }
        private static void PrintMismatches(List<KeyValuePair<int, string>> res, int wl, bool ws, bool wa)
        {
            // wl - write length, ws - write symbols, wa - write apart
            foreach (var line in res)
            {
                var length = wl <= -1 || wl > line.Value.Length ? line.Value.Length : wl;

                if (line.Key == -1)
                {
                    Console.Write("EOF \t\t:");
                    for (var i = 0; i < length; i++)
                    {
                        if (ws)
                        {
                            if (line.Value[i] > 33 && line.Value[i] < 127)
                                Console.Write(line.Value[i]);
                            else Console.Write('.');
                        }
                        else
                        {
                            if (line.Value[i] > 33 && line.Value[i] < 127)
                                Console.Write("0x{0:X} ", (int) line.Value[i]);
                            else Console.Write("0x{0:X} ", (int) '.');
                        }
                    }

                    break;
                }

                Console.Write("0x000{0:X} \t:", line.Key);
                
                if (wa)
                {
                    if (ws)                    
                        for (int i = 0, j = 1; j < length; i += 2, j += 2)
                            Console.Write("{0}({1})", line.Value[i], line.Value[j]);
                    else
                        for (int i = 0, j = 1; j < length; i += 2, j += 2)
                            Console.Write("0x{0:X}(0x{1:X})", (int)line.Value[i], (int)line.Value[j]);
                }
                else
                {
                    for (int i = 0; i < length; i += 2)
                        if (ws)
                            Console.Write("{0}", line.Value[i]);
                        else
                            Console.Write("0x{0:X} ", (int) line.Value[i]);


                    Console.Write("|");
                    for (int i = 1; i < length; i += 2)
                        if (ws)
                            Console.Write("{0}", line.Value[i]);
                        else
                            Console.Write("0x{0:X} ", (int) line.Value[i]);
                }

                Console.WriteLine();
            }
        }


        public static void Main(string[] args)
        {
            string file1, file2;
            bool showBrief = false,
                wrtSym = false,
                wrtApart = false;     
            int maxLen = -1;
            
            var parser = new OptionSet()
                .Add("l|len=", (int n) =>
                    {
                        if (n >= 0) maxLen = n;
                    })
                .Add("a|text", v => wrtSym = v!=null)
                .Add("y|side-by-side", v => wrtApart = v!=null )
                .Add("q|brief", v => showBrief = v!=null);
            parser.Parse(args);

            //Reading from files
            using (var fs = new StreamReader(args[0]))
            { file1 = fs.ReadToEnd(); }
            using (var fs = new StreamReader(args[1]))
            { file2 = fs.ReadToEnd(); }

            Console.WriteLine("File comparison");
            
            if(file1 == file2)
                Console.WriteLine("Files are identical.");
            else
            {
                Console.Write("Files are different");
                if (showBrief) Console.WriteLine(".");    // --q option
                else
                {
                    Console.WriteLine(":");
                    var result = Analyze(ref file1, ref file2);             
                    PrintMismatches(result, maxLen, wrtSym, wrtApart);
                }
            }            
        }
    }
}