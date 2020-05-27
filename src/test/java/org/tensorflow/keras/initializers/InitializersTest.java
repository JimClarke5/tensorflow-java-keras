/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/
package org.tensorflow.keras.initializers;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.function.Supplier;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.EagerSession;
import org.tensorflow.op.Ops;

/**
 *
 * @author Jim Clarke
 */
public class InitializersTest {

    public InitializersTest() {
    }

    @BeforeAll
    public static void setUpClass() {
    }

    @AfterAll
    public static void tearDownClass() {
    }

    @BeforeEach
    public void setUp() {
    }

    @AfterEach
    public void tearDown() {
    }

    /**
     * Test of get method, of class Initializers.
     */
    @Test
    public void testGet_Object_String() {
        System.out.println("get");
        String initializerFunction = "identity";
        try ( EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Initializer result = Initializers.get(tf, initializerFunction);
            assertNotNull(result);
            assertTrue(result instanceof Identity);
        }
    }

    /**
     * Test of get method, of class Initializers.
     */
    @Test
    public void testGet_Object_Lambda() {
        System.out.println("get");
        try ( EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Initializer result = Initializers.get(tf, ops -> new HeNormal(ops));
            assertNotNull(result);
            assertTrue(result instanceof HeNormal);
        }
    }

    /**
     * Test of get method, of class Initializers.
     */
    @Test
    public void testGet_Object_Class() {
        System.out.println("get");
        try ( EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Initializer result = Initializers.get(tf, Ones.class);
            assertNotNull(result);
            assertTrue(result instanceof Ones);
        }
    }

    /**
     * Test of get method, of class Initializers.
     */
    @Test
    public void testGet_Object_Initializer() {
        System.out.println("get");
        try ( EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Function<Ops, Initializer> initializerFunction = ops -> new Zeros(ops);
            Initializer result = Initializers.get(tf, initializerFunction);
            assertNotNull(result);
            assertTrue(result instanceof Zeros);
        }
    }

    /**
     * Test of get method, of class Initializers.
     */
    @Test
    public void testGet_Object_Unknown() {
        System.out.println("get");
        try ( EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            String initializerFunction = "bogus";
            Initializer result = Initializers.get(tf, initializerFunction);
            assertNull(result);
        }
    }

    /**
     * Test of get method, of class Initializers.
     */
    @Test
    public void testGet_Object_Map() {
        System.out.println("get");
        try ( EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Map<String, Function<Ops, Initializer>> custom_functions = new HashMap<String, Function<Ops, Initializer>>();
            custom_functions.put("foobar", ops -> new TruncatedNormal(ops));
            String initializerFunction = "foobar";
            Initializer result = Initializers.get(tf, initializerFunction, custom_functions);
            assertNotNull(result);
            assertTrue(result instanceof TruncatedNormal);
        }
    }

}
