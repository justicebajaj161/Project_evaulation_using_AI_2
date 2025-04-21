import Link from 'next/link';

export default function Navbar() {
  return (
    <nav className="p-4">
      <div className="container mx-auto flex justify-between items-center">
        <Link href="/" className="text-xl font-bold">
          Project Evaluator
        </Link>
        
        <div className="flex space-x-4">
          <Link href="/admin" className="hover:underline">
            Admin
          </Link>
          <Link href="/submit" className="hover:underline">
            Submit
          </Link>
          <Link href="/results" className="hover:underline">
            Results
          </Link>
        </div>
      </div>
    </nav>
  );
}