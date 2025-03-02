import unittest

from tracing_ml.parsing.ast.java import load_ast


class TestParseJava(unittest.TestCase):
    def test_something(self):
        for t in load_ast("""
        package com.hayden.ok;
       
        @Whatever 
        class Okay {
            private String okay; 
            
            public void doSomething() {
                throw new RuntimeException();
            }
        }
        """):
            print(t)


if __name__ == '__main__':
    unittest.main()
