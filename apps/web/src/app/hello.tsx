'use client';

import axios from "axios";

export default function Hello() {
	axios.get('http://localhost:3001').then((res) => {
		console.log(res);
	});
	return <div>Hello</div>;
}
