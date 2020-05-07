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
package org.tensorflow.keras.activations;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 *
 * @author Jim Clarke
 */
public class ActivationsTest {
    private static final double EPSILON = 1e-7;
    private static final float EPSILON_F = 1e-7f;
    
    public ActivationsTest() {
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
     * Test of get method, of class Activations.
     */
    @Test
    public void testGet_Object_String() {
        System.out.println("get");
        String  initializerFunction = "relu";
        Activation result = Activations.get(initializerFunction);
        assertNotNull(result);
        assertTrue(result instanceof ReLU);
    }
    
    /**
     * Test of get method, of class Activations.
     */
    @Test
    public void testGet_Object_Lambda() {
        System.out.println("get");
        Activation result = Activations.get(ReLU::new);
        assertNotNull(result);
        assertTrue(result instanceof ReLU);
    }
    
    /**
     * Test of get method, of class Activations.
     */
    @Test
    public void testGet_Object_Class() {
        System.out.println("get");
        Activation result = Activations.get(ReLU.class);
        assertNotNull(result);
        assertTrue(result instanceof ReLU);
    }
    
    /**
     * Test of get method, of class Activations.
     */
    @Test
    public void testGet_Object_Initializer() {
        System.out.println("get");
        ReLU initializerFunction =new ReLU();
        Activation result = Activations.get(initializerFunction);
        assertNotNull(result);
        assertTrue(result instanceof ReLU);
    }
    
    /**
     * Test of get method, of class Activations.
     */
    @Test
    public void testGet_Object_Unknown() {
        System.out.println("get");
        String initializerFunction = "bogus";
        Activation result = Activations.get(initializerFunction);
        assertNull(result);
    }

    /**
     * Test of get method, of class Activations.
     */
    @Test
    public void testGet_Object_Map() {
        System.out.println("get");
        Map<String, Supplier<Activation> > custom_functions = new HashMap<String, Supplier<Activation> >();
        custom_functions.put("foobar",ReLU::new);
        String initializerFunction = "foobar";
        Activation result = Activations.get(initializerFunction, custom_functions);
        assertNotNull(result);
        assertTrue(result instanceof ReLU);
    }

   
    
}
