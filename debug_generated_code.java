import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class Calculator {
    private final Lock lock = new ReentrantLock();

    public int add(int a, int b) {
        lock.lock();
        try {
            return a + b;
        } finally {
            lock.unlock();
        }
    }

    public int subtract(int a, int b) {
        lock.lock();
        try {
            return a - b;
        } finally {
            lock.unlock();
        }
    }

    public int multiply(int a, int b) {
        lock.lock();
        try {
            return a * b;
        } finally {
            lock.unlock();
        }
    }

    public double divide(int a, int b) {
        if (b == 0) {
            throw new ArithmeticException("Cannot divide by zero");
        }
        lock.lock();
        try {
            return a / b;
        } finally {
            lock.unlock();
        }
    }
}