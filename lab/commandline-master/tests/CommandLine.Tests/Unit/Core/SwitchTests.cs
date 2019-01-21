﻿// Copyright 2005-2015 Giacomo Stelluti Scala & Contributors. All rights reserved. See License.md in the project root for license information.

using System.Linq;
using CommandLine.Core;
using CSharpx;
using Xunit;
using FluentAssertions;

namespace CommandLine.Tests.Unit.Core
{
    public class SwitchTests
    {
        [Fact]
        public void Partition_switch_values_from_empty_token_sequence()
        {
            var expected = new Token[] { };

            var result = Switch.Partition(
                new Token[] { },
                name =>
                    new[] { "x", "switch" }.Contains(name)
                        ? Maybe.Just(TypeDescriptor.Create(TargetType.Switch, Maybe.Nothing<int>()))
                        : Maybe.Nothing<TypeDescriptor>());

            expected.Should().BeEquivalentTo(result);
        }

        [Fact]
        public void Partition_switch_values()
        {
            var expected = new [] { Token.Name("x") };

            var result = Switch.Partition(
                new []
                    {
                        Token.Name("str"), Token.Value("strvalue"), Token.Value("freevalue"),
                        Token.Name("x"), Token.Value("freevalue2")
                    },
                name =>
                    new[] { "x", "switch" }.Contains(name)
                        ? Maybe.Just(TypeDescriptor.Create(TargetType.Switch, Maybe.Nothing<int>()))
                        : Maybe.Nothing<TypeDescriptor>());

            expected.Should().BeEquivalentTo(result);
        }
    }
}
