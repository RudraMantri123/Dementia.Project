import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorCount: 0,
    };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);

    this.setState(prevState => ({
      error,
      errorInfo,
      errorCount: prevState.errorCount + 1,
    }));
  }

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });

    if (this.state.errorCount > 3) {
      window.location.reload();
    }
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 flex items-center justify-center p-4">
          <div className="max-w-2xl w-full bg-white dark:bg-gray-800 rounded-2xl shadow-2xl p-8">
            <div className="text-center">
              <div className="inline-flex items-center justify-center w-20 h-20 bg-red-100 dark:bg-red-900/30 rounded-full mb-6">
                <svg
                  className="w-10 h-10 text-red-600 dark:text-red-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                  />
                </svg>
              </div>

              <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
                Oops! Something went wrong
              </h1>

              <p className="text-lg text-gray-600 dark:text-gray-300 mb-8">
                We encountered an unexpected error. Don't worry, your data is safe.
              </p>

              {process.env.NODE_ENV === 'development' && this.state.error && (
                <div className="mb-8 text-left">
                  <details className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 cursor-pointer">
                    <summary className="font-semibold text-gray-700 dark:text-gray-300 mb-2">
                      Error Details (Development Only)
                    </summary>
                    <pre className="text-xs text-red-600 dark:text-red-400 overflow-auto max-h-40">
                      {this.state.error.toString()}
                      {'\n\n'}
                      {this.state.errorInfo?.componentStack}
                    </pre>
                  </details>
                </div>
              )}

              <div className="flex gap-4 justify-center">
                <button
                  onClick={this.handleReset}
                  className="px-6 py-3 bg-blue-600 dark:bg-blue-500 text-white font-semibold rounded-lg hover:bg-blue-700 dark:hover:bg-blue-600 transition-colors shadow-md"
                >
                  Try Again
                </button>

                <button
                  onClick={() => window.location.href = '/'}
                  className="px-6 py-3 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 font-semibold rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors shadow-md"
                >
                  Go Home
                </button>
              </div>

              <div className="mt-8 text-sm text-gray-500 dark:text-gray-400">
                {this.state.errorCount > 2 && (
                  <p className="text-orange-600 dark:text-orange-400 font-semibold mb-2">
                    Multiple errors detected. Consider refreshing the page.
                  </p>
                )}
                <p>
                  If the problem persists, please contact support or try refreshing the
                  page.
                </p>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
