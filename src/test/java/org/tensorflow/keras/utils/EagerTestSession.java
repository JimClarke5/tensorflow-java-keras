/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.utils;

import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.tensorflow.DataType;
import org.tensorflow.EagerSession;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.family.TNumber;

/**
 *
 * @author Jim Clarke
 */
public class EagerTestSession extends TestSession {
    private final EagerSession session;
    private final Ops tf;
    
    
    
    public EagerTestSession() {
        this.session = EagerSession.create();
        this.tf = Ops.create(session).withName("test");
    }
    
    public Ops getTF() {
        return tf;
    }
    
     public EagerSession getSession() {
        return session;
    }

    @Override
    public void close()  {
        session.close();
    }

    @Override
    public boolean isEager() {
        return true;
    }

    @Override
    public Session getGraphSession() {
        return null;
    }

    @Override
    public EagerSession getEagerSession() {
        return this.session;
    }

    @Override
    public <T extends TNumber> void evaluate(Number[] expected, Operand<T> input) {
        boolean scalarExpected = expected.length == 1;
        DataType dtype = input.asOutput().dataType();
        if(dtype == TFloat32.DTYPE) {
            Operand<TFloat32> o = (Operand<TFloat32>)input;
            AtomicInteger index = new AtomicInteger();
            o.data().scalars().forEach(f -> {
                   assertEquals(expected[index.get()].floatValue(), f.getFloat(), epsilon);
                   if(!scalarExpected)
                       index.incrementAndGet();
                       
            });
        }else if(dtype == TFloat64.DTYPE) {
            Operand<TFloat64> o = (Operand<TFloat64>)input;
            AtomicInteger index = new AtomicInteger();
            o.data().scalars().forEach(f -> {
                   assertEquals(expected[index.get()].doubleValue(), f.getDouble(), epsilon);
                   if(!scalarExpected)
                       index.incrementAndGet();
            });
        }else if(dtype == TInt32.DTYPE) {
            Operand<TInt32> o = (Operand<TInt32>)input;
            AtomicInteger index = new AtomicInteger();
            o.data().scalars().forEach(f -> {
                   assertEquals(expected[index.get()].intValue(), f.getInt());
                   if(!scalarExpected)
                       index.incrementAndGet();
            });
        } else if(dtype == TInt64.DTYPE) {
            Operand<TInt64> o = (Operand<TInt64>)input;
            AtomicInteger index = new AtomicInteger();
            o.data().scalars().forEach(f -> {
                   assertEquals(expected[index.get()].longValue(), f.getLong());
                   if(!scalarExpected)
                       index.incrementAndGet();
            });
        }
        
    }
}
