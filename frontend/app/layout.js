import Navbar from '@/components/Navbar';
import './globals.css';

export default function RootLayout({ children }) {
  return (
    <html lang="en" data-theme="valentine">
      <body>
        <Navbar />
        <main className="min-h-screen bg-gray-50 py-8">
          {children}
        </main>
      </body>
    </html>
  );
}